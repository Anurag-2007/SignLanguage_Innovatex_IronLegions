import React, { useEffect, useRef, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as handPoseDetection from "@tensorflow-models/hand-pose-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

export default function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [gesture, setGesture] = useState("⏳ Preparing Model...");
  const [text, setText] = useState("");
  const [dim, setDim] = useState({ w: 640, h: 480 });

  const lastWrittenLetter = useRef("");
  const stableGesture = useRef("");
  const stableCount = useRef(0);
  const STABLE_THRESHOLD = 15;

  // --- SMOOTHING BUFFER ---
  const gestureBuffer = useRef([]); 
  const BUFFER_SIZE = 5;

  // --- WAVE VARS ---
  const prevWristX = useRef(null);
  const waveCycles = useRef(0);
  const waveDirection = useRef(0);
  const stationaryFrames = useRef(0);

  // Theme constants
  const THEME_COLOR = "#FFFFC5";
  const THEME_RGB = "255, 255, 197";

  useEffect(() => {
    const updateDimensions = () => {
      const maxWidth = window.innerWidth - 40;
      const maxHeight = window.innerHeight * 0.5;
      const aspectRatio = 4 / 3;
      let width = maxWidth;
      let height = width / aspectRatio;
      if (height > maxHeight) {
        height = maxHeight;
        width = height * aspectRatio;
      }
      setDim({ w: Math.round(width), h: Math.round(height) });
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
      setGesture("👀 Show your hand");
    })();
  }, []);

  const recognizeGesture = useCallback((hand) => {
    const k = hand.keypoints;
    const handSize = Math.hypot(k[9].x - k[0].x, k[9].y - k[0].y);
    const T = (factor) => handSize * factor; 

    // --- 1. DETERMINE FINGER STATES ---
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

    // --- 2. WAVE DETECTION (Strict) ---
    if (indexExt && middleExt && ringExt && pinkyExt) {
        const wristX = k[0].x;
        if (prevWristX.current !== null) {
            const diff = wristX - prevWristX.current;
            const speed = Math.abs(diff);

            if (speed > T(0.04)) {
                stationaryFrames.current = 0;
                const currentDir = diff > 0 ? 1 : -1;
                if (currentDir !== waveDirection.current) {
                    waveCycles.current += 1;
                    waveDirection.current = currentDir;
                }
            } else {
                stationaryFrames.current += 1;
                if (stationaryFrames.current > 10) {
                    waveCycles.current = 0;
                }
            }
        }
        prevWristX.current = wristX;
    } else {
        waveCycles.current = 0;
        prevWristX.current = null;
    }

    if (waveCycles.current >= 4) return "👋 HELLO";

    // --- 3. SINGLE FINGER GROUP (D, G, L, X) ---
    // This handles ALL cases where Index is Extended but others are Curled.
    if (indexExt && !middleExt && !ringExt && !pinkyExt) {
        
        // ** G Check (Horizontal) **
        // Compare X width vs Y height of the Index Finger.
        // If Width > Height, finger is sideways.
        const xDiff = Math.abs(k[8].x - k[5].x);
        const yDiff = Math.abs(k[8].y - k[5].y);
        
        // T(0.1) adds a buffer so slightly tilted fingers don't flicker.
        if (xDiff > yDiff + T(0.1)) {
            return "G";
        }

        // ** L Check (Thumb Out) **
        const distThumbBase = Math.hypot(thumbTip.x - indexMcp.x, thumbTip.y - indexMcp.y);
        if (distThumbBase > T(0.9)) {
            return "L";
        }
        
        // ** X Check (Hooked) **
        // If the tip is lower than the mid-joint (PIP), it's hooked.
        // Note: Y increases downwards.
        if (k[8].y > k[6].y - T(0.2)) {
            return "X";
        }

        // ** D Check (Vertical) **
        // If it's not G, L, or X, it must be D.
        return "D";
    }

    // --- 4. GLOBAL "O" CHECK ---
    if (!middleExt && !ringExt && !pinkyExt) {
        if (distIndexTipKnuckle > T(0.55)) return "O";
    }

    // --- 5. STRICT F CHECK ---
    if (distThumbIndex < T(0.45)) { 
       if (middleExt && ringExt && pinkyExt) return "F";
    }

    // --- 6. TWO FINGER GROUP (H, U, R, V) ---
    if (indexExt && middleExt && !ringExt && !pinkyExt) {
       const xDiff = Math.abs(k[8].x - k[5].x);
       const yDiff = Math.abs(k[8].y - k[5].y);
       if (xDiff > yDiff) return "H";

       if (k[8].x > k[12].x && k[8].x - k[12].x < T(0.4)) return "R"; 
       const distTips = Math.hypot(k[8].x - k[12].x, k[8].y - k[12].y);
       if (distTips > T(0.5)) return "V";
       return "U";
    }

    // --- 7. FIST GROUP (A, E, S, T) ---
    if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
      if (distIndexTipKnuckle > T(0.65)) return "O"; 
      if (thumbTip.y > indexMcp.y) return "E"; 
      const xDist = Math.abs(thumbTip.x - indexMcp.x);
      if (xDist > T(0.25)) return "A"; 
      return "S"; 
    }

    // --- 8. PINKY ONLY GROUP (I, Y) ---
    if (!indexExt && !middleExt && !ringExt && pinkyExt) {
       const spread = Math.hypot(thumbTip.x - k[20].x, thumbTip.y - k[20].y);
       if (spread > T(1.2)) return "Y";
       return "I";
    }

    // --- 9. OPEN HAND VARIANTS (B, C, Neutral) ---
    if (indexExt && middleExt && ringExt && pinkyExt) {
       if (Math.abs(thumbTip.x - indexMcp.x) < T(0.35)) return "B";
       if (distThumbIndex < T(0.8) && distIndexTipKnuckle < T(0.85)) return "C";
       return "🖐️"; 
    }
    
    // W Check
    if (indexExt && middleExt && ringExt && !pinkyExt) return "W";

    return "🖐️";
  }, []);

  const detect = useCallback(async () => {
    if (!detector || !webcamRef.current?.video) return;
    const video = webcamRef.current.video;
    if (video.readyState !== 4) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const hands = await detector.estimateHands(video, { flipHorizontal: true });

    if (hands.length > 0) {
      const hand = hands[0];
      const rawGesture = recognizeGesture(hand);

      // --- SMOOTHING ---
      gestureBuffer.current.push(rawGesture);
      if (gestureBuffer.current.length > BUFFER_SIZE) {
          gestureBuffer.current.shift();
      }

      const counts = {};
      let maxCount = 0;
      let smoothedGesture = rawGesture;

      gestureBuffer.current.forEach(g => {
          counts[g] = (counts[g] || 0) + 1;
          if (counts[g] > maxCount) {
              maxCount = counts[g];
              smoothedGesture = g;
          }
      });

      setGesture(smoothedGesture);

      // --- TYPING ---
      if (smoothedGesture === "👋 HELLO") {
          if (lastWrittenLetter.current !== "👋 HELLO") {
               setText(t => t + " HELLO ");
               lastWrittenLetter.current = "👋 HELLO";
               waveCycles.current = 0;
          }
      } 
      else {
          if (smoothedGesture === stableGesture.current) {
            stableCount.current += 1;
          } else {
            stableGesture.current = smoothedGesture;
            stableCount.current = 1;
          }

          if (stableCount.current >= STABLE_THRESHOLD) {
             if (smoothedGesture !== "🖐️" && smoothedGesture !== "👀 Show your hand") {
                 if (smoothedGesture !== lastWrittenLetter.current) {
                     setText(t => t + smoothedGesture);
                     lastWrittenLetter.current = smoothedGesture;
                 }
             } else if (smoothedGesture === "🖐️") {
                 lastWrittenLetter.current = "";
             }
          }
      }

      ctx.fillStyle = THEME_COLOR;
      hand.keypoints.forEach(p => {
         ctx.beginPath();
         ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
         ctx.fill();
      });
    } else {
      setGesture("👀 Show your hand");
      stableCount.current = 0;
      lastWrittenLetter.current = "";
    }
  }, [detector, recognizeGesture]);

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
    <div style={{ background: "#ede7e7ff", minHeight: "100vh", color: "#210303ff", textAlign: "center", padding: 10 }}>
      <h1 style={{ fontSize: "clamp(2rem, 6vw, 3rem)" }}>🤟 ASL Fingerspelling</h1>
      
      <h2 style={{ fontSize: "clamp(3rem, 10vw, 5rem)", color: "#272704ff", margin: "10px 0" }}>{gesture}</h2>
      
      <div style={{
        margin: "10px auto", padding: 15, 
        border: `2px solid ${THEME_COLOR}`, 
        borderRadius: 12,
        minHeight: 80, maxWidth: "90%", fontSize: "1.6rem", 
        background: `rgba(${THEME_RGB}, 0.1)`, 
        wordWrap: "break-word"
      }}>
        {text || "✍️ Start signing..."}
      </div>

      <div style={{ display: "flex", justifyContent: "center", gap: "20px", margin: "15px 0" }}>
        <button
          onClick={() => { setText(""); lastWrittenLetter.current = ""; }}
          style={{ padding: "12px 24px", background: "red", color: "#fff", border: "none", borderRadius: 10, fontSize: "1.2rem", fontWeight: "bold", cursor: "pointer" }}
        >
          CLEAR ALL
        </button>
        <button
          onClick={() => { setText(t => t.slice(0, -1)); lastWrittenLetter.current = ""; }}
          style={{ padding: "12px 24px", background: "orange", color: "#fff", border: "none", borderRadius: 10, fontSize: "1.2rem", fontWeight: "bold", cursor: "pointer" }}
        >
          ⌫ BACKSPACE
        </button>
      </div>

      <div style={{
        margin: "auto", width: dim.w, height: dim.h, 
        border: `6px solid #272704ff`,
        borderRadius: "20px",
        position: "relative", 
        boxShadow: `0 0 30px #010000ff`, 
        overflow: "hidden"
      }}>
        <Webcam ref={webcamRef} mirrored width={dim.w} height={dim.h} videoConstraints={{ width: dim.w, height: dim.h, facingMode: "user" }} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }} />
      </div>
    </div>
  );
}