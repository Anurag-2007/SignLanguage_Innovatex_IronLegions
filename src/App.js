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
  const [dim, setDim] = useState({ w: 480, h: 360 });

  const lastWrittenLetter = useRef("");
  const stableGesture = useRef("");
  const stableCount = useRef(0);
  const STABLE_THRESHOLD = 15;

  // --- SPACEBAR LOGIC VAR ---
  const wasHandPresent = useRef(false);

  // --- SMOOTHING BUFFER ---
  const gestureBuffer = useRef([]);
  const BUFFER_SIZE = 10;

  const THEME_COLOR = "#FFFFC5";
  const THEME_RGB = "255, 255, 197";

  // --- KEYBOARD LISTENERS ---
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Backspace") {
        setText((prev) => prev.slice(0, -1));
        lastWrittenLetter.current = "";
      } else if (event.key === "Escape") {
        setText("");
        lastWrittenLetter.current = "";
      } else if (event.key === " ") {
        // Spacebar support
        setText((prev) => prev + " ");
        lastWrittenLetter.current = "";
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // --- RESPONSIVE DIMENSIONS ---
  useEffect(() => {
    const updateDimensions = () => {
      const screenW = window.innerWidth;
      const screenH = window.innerHeight;
      const isPortrait = screenH > screenW;

      let w, h;

      if (isPortrait) {
        // Mobile (Portrait)
        w = Math.min(screenW - 20, 480);
        h = w * 1.333; 
      } else {
        // Desktop (Landscape)
        w = Math.min(screenW - 40, 480); 
        h = w * 0.75; 
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
    if (indexExt && middleExt && ringExt && pinkyExt) {
      if (Math.abs(thumbTip.x - indexMcp.x) < T(0.35)) return "B";
      if (distThumbIndex < T(0.8) && distIndexTipKnuckle < T(0.85)) return "C";
      return "🖐️";
    }

    if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
      if (distIndexTipKnuckle > T(0.65)) return "O";
      const thumbX = thumbTip.x;
      const indexX = indexMcp.x;
      const thumbY = thumbTip.y;
      const indexY = indexMcp.y;

      const distThumbMiddleX = Math.abs(thumbX - k[9].x);
      if (distThumbMiddleX < T(0.2)) return "S";

      const xDist = Math.abs(thumbX - indexX);
      if (xDist > T(0.35)) return "A";

      const thumbToIndexDist = Math.hypot(thumbX - indexX, thumbY - indexY);
      if (thumbToIndexDist < T(0.4)) return "E";

      if (Math.abs(thumbX - k[17].x) < T(0.3)) return "M"; 
      if (Math.abs(thumbX - k[13].x) < T(0.3)) return "M"; 
      if (Math.abs(thumbX - k[9].x) < T(0.3))  return "N"; 

      return "✊"; 
    }

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

    if (!indexExt && !middleExt && !ringExt && pinkyExt) {
       const spread = Math.hypot(thumbTip.x - k[20].x, thumbTip.y - k[20].y);
       if (spread > T(1.2)) return "Y";
       return "I";
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
    
    canvas.width = dim.w;
    canvas.height = dim.h;
    ctx.clearRect(0, 0, dim.w, dim.h);

    const hands = await detector.estimateHands(video, { flipHorizontal: true });

    if (hands.length > 0) {
      // ** HAND DETECTED **
      wasHandPresent.current = true; // Mark hand as present

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
        if (smoothedGesture !== "🖐️" && smoothedGesture !== "👀 Show your hand" && smoothedGesture !== "✊") {
          if (smoothedGesture !== lastWrittenLetter.current) {
            setText((t) => t + smoothedGesture);
            lastWrittenLetter.current = smoothedGesture;
          }
        } else if (smoothedGesture === "🖐️" || smoothedGesture === "✊") {
          lastWrittenLetter.current = ""; 
        }
      }

      ctx.fillStyle = THEME_COLOR;
      hand.keypoints.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
      });
    } else {
      // ** HAND LOST (Auto Space Logic) **
      setGesture("👀 Show Hand");
      stableCount.current = 0;
      lastWrittenLetter.current = "";

      if (wasHandPresent.current) {
          // If the hand was visible and now it's gone, add a space
          setText(prev => (prev.length > 0 && !prev.endsWith(" ") ? prev + " " : prev));
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
        justifyContent: "flex-start",
        padding: "15px",
        boxSizing: "border-box",
        overflowX: "hidden"
      }}
    >
      <h1 style={{ fontSize: "clamp(0.8rem, 4vw, 1rem)", margin: "0 0 10px 0" }}>
        🤟 ASL Fingerspelling
      </h1>

      <h2 style={{ fontSize: "clamp(1rem, 6vw, 1.5rem)", margin: "0 0 10px 0", color: "#272704ff", minHeight: "30px" }}>
        {gesture}
      </h2>

      <div
        style={{
          width: "100%",
          maxWidth: "480px",
          padding: "10px",
          border: `2px solid ${THEME_COLOR}`,
          borderRadius: 12,
          minHeight: "50px",
          fontSize: "1.2rem",
          background: `rgba(${THEME_RGB}, 0.1)`,
          wordWrap: "break-word",
          textAlign: "left",
          marginBottom: "15px"
        }}
      >
        {text || <span style={{opacity:0.5}}>Start signing...</span>}
      </div>

      <div style={{ display: "flex", gap: "15px", marginBottom: "15px" }}>
        <button
          onClick={() => { setText(""); lastWrittenLetter.current = ""; }}
          style={{
            padding: "8px 16px",
            background: "red",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            fontSize: "0.9rem",
            fontWeight: "bold",
            cursor: "pointer"
          }}
        >
          CLEAR
        </button>

        <button
          onClick={() => { setText((t) => t.slice(0, -1)); lastWrittenLetter.current = ""; }}
          style={{
            padding: "8px 16px",
            background: "orange",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            fontSize: "0.9rem",
            fontWeight: "bold",
            cursor: "pointer"
          }}
        >
          BACK
        </button>
      </div>

      <div
        style={{
          position: "relative",
          width: dim.w,
          height: dim.h,
          border: `4px solid #272704ff`,
          borderRadius: "20px",
          overflow: "hidden",
          backgroundColor: "#000",
          boxShadow: `0 0 15px rgba(0,0,0,0.3)`
        }}
      >
        <Webcam
          ref={webcamRef}
          mirrored
          width={dim.w}
          height={dim.h}
          videoConstraints={{
            facingMode: "user",
            aspectRatio: dim.h > dim.w ? 0.75 : 1.333
          }}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
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