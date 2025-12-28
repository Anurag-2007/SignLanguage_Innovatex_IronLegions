import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [gesture, setGesture] = useState("⏳ Loading...");
  const [text, setText] = useState("");
  const [dim, setDim] = useState({ w: 640, h: 480 });

  const lastWrittenLetter = useRef("");
  const stableGesture = useRef("");
  const stableCount = useRef(0);
  const STABLE_THRESHOLD = 15;

  // --- Track Hand Presence for Auto-Space ---
  const wasHandPresent = useRef(false);

  // --- SMOOTHING BUFFER ---
  const gestureBuffer = useRef([]);
  const BUFFER_SIZE = 5;

  // Theme constants
  const THEME_COLOR = "#FFFFC5";
  const THEME_RGB = "255, 255, 197";

  // --- RESPONSIVE DIMENSIONS LOGIC ---
  useEffect(() => {
    const updateDimensions = () => {
      // Get available screen width (minus some padding)
      const screenWidth = window.innerWidth - 20; 
      // Limit max height to leave room for text/buttons on mobile
      const maxScreenHeight = window.innerHeight * 0.6; 
      
      const targetAspectRatio = 4 / 3;
      
      let w = screenWidth;
      let h = w / targetAspectRatio;

      // If calculated height is too tall, scale down based on height
      if (h > maxScreenHeight) {
        h = maxScreenHeight;
        w = h * targetAspectRatio;
      }

      // Cap at 640x480 for desktop performance
      if (w > 640) {
        w = 640;
        h = 480;
      }

      setDim({ w: Math.round(w), h: Math.round(h) });
    };

    updateDimensions();
    window.addEventListener("resize", updateDimensions);
    window.addEventListener("orientationchange", updateDimensions);
    return () => {
      window.removeEventListener("resize", updateDimensions);
      window.removeEventListener("orientationchange", updateDimensions);
    };
  }, []);

  useEffect(() => {
    (async () => {
      await tf.ready();
      await tf.setBackend("webgl");
      const d = await handPoseDetection.createDetector(
        handPoseDetection.SupportedModels.MediaPipeHands,
        { runtime: "tfjs", modelType: "full", maxHands: 1 }
      );
      setDetector(d);
      setGesture("👀 Show Hand");
    })();
  }, []);

  const recognizeGesture = useCallback((hand) => {
    const k = hand.keypoints;
    const handSize = Math.hypot(k[9].x - k[0].x, k[9].y - k[0].y);
    const T = (factor) => handSize * factor;

    const isExtended = (tip, mcp) => {
      const dist = Math.hypot(k[tip].x - k[mcp].x, k[tip].y - k[mcp].y);
      return dist > T(0.6);
    };

    const indexExt = isExtended(8, 5);
    const middleExt = isExtended(12, 9);
    const ringExt = isExtended(16, 13);
    const pinkyExt = isExtended(20, 17);

    const thumbTip = k[4];
    const indexMcp = k[5];
    const distThumbIndex = Math.hypot(thumbTip.x - k[8].x, thumbTip.y - k[8].y);
    const distIndexTipKnuckle = Math.hypot(k[8].x - k[5].x, k[8].y - k[5].y);

    // --- LOGIC ---
    if (indexExt && !middleExt && !ringExt && !pinkyExt) {
      const xDiff = Math.abs(k[8].x - k[5].x);
      const yDiff = Math.abs(k[8].y - k[5].y);
      if (xDiff > yDiff + T(0.1)) return "G";
      const distThumbBase = Math.hypot(thumbTip.x - indexMcp.x, thumbTip.y - indexMcp.y);
      if (distThumbBase > T(0.9)) return "L";
      if (k[8].y > k[6].y - T(0.2)) return "X";
      return "D";
    }

    if (!middleExt && !ringExt && !pinkyExt) {
      if (distIndexTipKnuckle > T(0.55)) return "O";
    }

    if (distThumbIndex < T(0.45)) {
      if (middleExt && ringExt && pinkyExt) return "F";
    }

    if (indexExt && middleExt && !ringExt && !pinkyExt) {
      const xDiff = Math.abs(k[8].x - k[5].x);
      const yDiff = Math.abs(k[8].y - k[5].y);
      if (xDiff > yDiff) return "H";
      if (k[8].x > k[12].x && k[8].x - k[12].x < T(0.4)) return "R";
      const distTips = Math.hypot(k[8].x - k[12].x, k[8].y - k[12].y);
      if (distTips > T(0.5)) return "V";
      return "U";
    }

    if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
      if (distIndexTipKnuckle > T(0.65)) return "O";
      if (thumbTip.y > indexMcp.y) return "E";
      const xDist = Math.abs(thumbTip.x - indexMcp.x);
      if (xDist > T(0.25)) return "A";
      return "S";
    }

    if (!indexExt && !middleExt && !ringExt && pinkyExt) {
      const spread = Math.hypot(thumbTip.x - k[20].x, thumbTip.y - k[20].y);
      if (spread > T(1.2)) return "Y";
      return "I";
    }

    if (indexExt && middleExt && ringExt && pinkyExt) {
      if (Math.abs(thumbTip.x - indexMcp.x) < T(0.35)) return "B";
      if (distThumbIndex < T(0.8) && distIndexTipKnuckle < T(0.85)) return "C";
      return "🖐️";
    }

    if (indexExt && middleExt && ringExt && !pinkyExt) return "W";

    return "🖐️";
  }, []);

  const detect = useCallback(async () => {
    if (!detector || !webcamRef.current?.video) return;
    const video = webcamRef.current.video;
    if (video.readyState !== 4) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    // Explicitly set canvas size to match the calculated dimension state
    canvas.width = dim.w;
    canvas.height = dim.h;
    ctx.clearRect(0, 0, dim.w, dim.h);

    const hands = await detector.estimateHands(video, { flipHorizontal: true });

    if (hands.length > 0) {
      wasHandPresent.current = true;
      const hand = hands[0];
      const rawGesture = recognizeGesture(hand);

      gestureBuffer.current.push(rawGesture);
      if (gestureBuffer.current.length > BUFFER_SIZE) {
        gestureBuffer.current.shift();
      }

      const counts = {};
      let maxCount = 0;
      let smoothedGesture = rawGesture;

      gestureBuffer.current.forEach((g) => {
        counts[g] = (counts[g] || 0) + 1;
        if (counts[g] > maxCount) {
          maxCount = counts[g];
          smoothedGesture = g;
        }
      });

      setGesture(smoothedGesture);

      if (smoothedGesture === stableGesture.current) {
        stableCount.current += 1;
      } else {
        stableGesture.current = smoothedGesture;
        stableCount.current = 1;
      }

      if (stableCount.current >= STABLE_THRESHOLD) {
        if (smoothedGesture !== "🖐️" && smoothedGesture !== "👀 Show your hand") {
          if (smoothedGesture !== lastWrittenLetter.current) {
            setText((t) => t + smoothedGesture);
            lastWrittenLetter.current = smoothedGesture;
          }
        } else if (smoothedGesture === "🖐️") {
          lastWrittenLetter.current = "";
        }
      }

      ctx.fillStyle = THEME_COLOR;
      hand.keypoints.forEach((p) => {
        ctx.beginPath();
        // Scale points if video size differs from internal resolution
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    } else {
      setGesture("👀 Show Hand");
      stableCount.current = 0;
      lastWrittenLetter.current = "";

      // ** AUTO SPACE LOGIC **
      if (wasHandPresent.current) {
          setText(prev => {
              if (prev.length > 0 && !prev.endsWith(" ")) {
                  return prev + " ";
              }
              return prev;
          });
          wasHandPresent.current = false;
      }
    }
  }, [detector, recognizeGesture, dim]);

  useEffect(() => {
    if (!detector) return;
    let raf;
    const loop = () => {
      detect();
      raf = requestAnimationFrame(loop);
    };
    loop();
    return () => cancelAnimationFrame(raf);
  }, [detector, detect]);

  return (
    <div
      style={{
        background: "#ede7e7ff",
        minHeight: "100vh",
        color: "#210303ff",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "flex-start", // Align top for mobile
        padding: "20px 10px", // Less padding on sides for mobile
        boxSizing: "border-box",
        overflowX: "hidden"
      }}
    >
      <h1 style={{ 
        fontSize: "clamp(1.8rem, 5vw, 2.5rem)", 
        margin: "0 0 10px 0" 
      }}>
        🤟 ASL Fingerspelling
      </h1>

      <h2
        style={{
          fontSize: "clamp(3rem, 10vw, 4.5rem)",
          color: "#272704ff",
          margin: "0 0 15px 0",
          minHeight: "60px"
        }}
      >
        {gesture}
      </h2>

      <div
        style={{
          width: "100%",
          maxWidth: "600px",
          padding: "15px",
          border: `2px solid ${THEME_COLOR}`,
          borderRadius: 12,
          minHeight: "70px",
          fontSize: "1.5rem",
          background: `rgba(${THEME_RGB}, 0.1)`,
          wordWrap: "break-word",
          textAlign: "left",
          marginBottom: "20px"
        }}
      >
        {text || <span style={{opacity: 0.6}}>✍️ Start signing...</span>}
      </div>

      <div
        style={{
          display: "flex",
          justifyContent: "center",
          gap: "15px",
          marginBottom: "20px",
          flexWrap: "wrap"
        }}
      >
        <button
          onClick={() => {
            setText("");
            lastWrittenLetter.current = "";
          }}
          style={{
            padding: "12px 24px",
            background: "red",
            color: "#fff",
            border: "none",
            borderRadius: 10,
            fontSize: "1rem",
            fontWeight: "bold",
            cursor: "pointer",
            flex: "1 1 auto",
            minWidth: "120px"
          }}
        >
          CLEAR ALL
        </button>

        <button
          onClick={() => {
            setText((t) => t.slice(0, -1));
            lastWrittenLetter.current = "";
          }}
          style={{
            padding: "12px 24px",
            background: "orange",
            color: "#fff",
            border: "none",
            borderRadius: 10,
            fontSize: "1rem",
            fontWeight: "bold",
            cursor: "pointer",
            flex: "1 1 auto",
            minWidth: "120px"
          }}
        >
          ⌫ BACKSPACE
        </button>
      </div>

      {/* Camera Container */}
      <div
        style={{
          position: "relative",
          width: dim.w,
          height: dim.h,
          border: `4px solid #272704ff`,
          borderRadius: "20px",
          boxShadow: `0 0 20px #010000ff`,
          overflow: "hidden",
          background: "#000"
        }}
      >
        <Webcam
          ref={webcamRef}
          mirrored
          width={dim.w}
          height={dim.h}
          videoConstraints={{
            width: dim.w,
            height: dim.h,
            facingMode: "user",
          }}
          style={{ 
            width: "100%", 
            height: "100%", 
            objectFit: "cover" // Ensures video fills the container
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
          }}
        />
      </div>
    </div>
  );
}