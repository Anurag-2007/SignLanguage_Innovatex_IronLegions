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

  const wasHandPresent = useRef(false);
  const gestureBuffer = useRef([]);
  const BUFFER_SIZE = 10;

  const THEME_COLOR = "#FFFFC5";
  const THEME_RGB = "255, 255, 197";

  // --- SETUP KEYBOARD & TTS ---
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === "Backspace") {
        setText((prev) => prev.slice(0, -1));
        lastWrittenLetter.current = "";
      } else if (event.key === "Escape") {
        setText("");
        lastWrittenLetter.current = "";
      } else if (event.key === " ") {
        setText((prev) => prev + " ");
        lastWrittenLetter.current = "";
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const speakText = () => {
    if (!text) return;
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1;
    window.speechSynthesis.speak(utterance);
  };

  // --- RESPONSIVE DIMENSIONS ---
  useEffect(() => {
    const updateDimensions = () => {
      const screenW = window.innerWidth;
      const screenH = window.innerHeight;
      const isPortrait = screenH > screenW;
      let w, h;
      if (isPortrait) {
        w = Math.min(screenW - 20, 480);
        h = w * 1.333;
      } else {
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

    // Finger Helpers
    // isExtended: Distance from tip to knuckle is large
    const isExtended = (tipIdx, mcpIdx) => {
        const dist = Math.hypot(k[tipIdx].x - k[mcpIdx].x, k[tipIdx].y - k[mcpIdx].y);
        return dist > T(0.55);
    };

    // isCurled: Tip is close to knuckle or lower than pip
    const isCurled = (tipIdx, pipIdx) => {
         const dist = Math.hypot(k[tipIdx].x - k[pipIdx].x, k[tipIdx].y - k[pipIdx].y);
         return dist < T(0.4);
    };

    const indexExt = isExtended(8, 5);
    const middleExt = isExtended(12, 9);
    const ringExt = isExtended(16, 13);
    const pinkyExt = isExtended(20, 17);

    // Orientation Check
    // If Index Tip Y is HIGHER than Wrist Y, hand is pointing DOWN (coordinates start 0 at top)
    const pointingDown = k[8].y > k[0].y; 
    
    const d = (i1, i2) => Math.hypot(k[i1].x - k[i2].x, k[i1].y - k[i2].y);

    // --- LOGIC START ---

    // 1. DOWNWARD GROUP (P, Q)
    if (pointingDown) {
        // Q: "Index and Thumb Open" pointing down
        if (indexExt && !middleExt && !ringExt && !pinkyExt) {
             if (d(4, 8) > T(0.6)) return "Q";
        }
        
        // P: "V shape pointed downwards and thumb tuck"
        // Index & Middle extended, Thumb between them
        if (indexExt && middleExt && !ringExt && !pinkyExt) {
             return "P"; 
        }
    }

    // 2. HORIZONTAL GROUP (G, H)
    const isHorizontal = Math.abs(k[8].x - k[5].x) > Math.abs(k[8].y - k[5].y) * 1.5;
    if (isHorizontal && !ringExt && !pinkyExt) {
        if (middleExt) return "H";
        return "G";
    }

    // 3. SINGLE FINGER GROUP (D, L, X, Z)
    if (!middleExt && !ringExt && !pinkyExt) {
        if (indexExt) {
             // L Check
             if (d(4, 5) > T(0.9)) return "L";
             return "D";
        }
        
        // X CHECK: "Index Hooked"
        // Index is NOT fully extended, but NOT fully curled like a fist
        // We check if Index Tip is somewhere in between
        const indexLen = d(8, 5);
        if (indexLen < T(0.5) && indexLen > T(0.2)) {
             // Thumb is usually tucked
             return "X";
        }
    }

    // 4. TWO FINGER GROUP (U, V, R, K)
    if (indexExt && middleExt && !ringExt && !pinkyExt) {
        if (Math.abs(k[8].x - k[12].x) < T(0.25)) return "R"; // Crossed

        // K CHECK: "V pointed upwards and thumb tuck in between"
        // Thumb Tip (4) should be close to the middle/index knuckles
        const thumbY = k[4].y;
        const middleKnuckleY = k[9].y;
        
        // If thumb tip is "high" (near knuckles)
        if (thumbY < middleKnuckleY + T(0.2)) {
             return "K";
        }

        if (d(8, 12) > T(0.45)) return "V";
        return "U";
    }

    // 5. OPEN HAND / W / F
    if (indexExt && middleExt && ringExt) {
        if (pinkyExt) {
            if (d(4, 17) < T(1.0)) return "B"; 
            return "🖐️";
        }
        return "W";
    }

    // F Check (Circle)
    if (!indexExt && middleExt && ringExt && pinkyExt) {
         if (d(4, 8) < T(0.5)) return "F";
    }

    // 6. PINKY GROUP (I, Y)
    if (!indexExt && !middleExt && !ringExt && pinkyExt) {
        if (d(4, 17) > T(1.1)) return "Y";
        return "I";
    }

    // 7. FIST GROUP (A, E, M, N, S, T, O)
    // All fingers curled/closed
    if (!indexExt && !middleExt && !ringExt && !pinkyExt) {
        
        // O Check (Thumb touches Index Tip)
        if (d(4, 8) < T(0.5)) return "O";

        // E Check (Thumb curled low, near palm center)
        if (d(4, 13) < T(0.5) && k[4].y > k[13].y) return "E";

        // A Check (Thumb on the side, sticking up)
        if (d(4, 5) > T(0.5) && k[4].y < k[5].y) return "A";
        
        // S Check (Thumb crosses OVER the fingers)
        // This is tricky to separate from M/N/T without depth.
        // S usually locks the fist, thumb crossing index & middle.
        
        // ** M, N, T LOGIC **
        // We look at how far the thumb reaches across the hand (X-axis)
        // We compare Thumb Tip X to the Knuckles (MCP) X
        
        // Find closest knuckle to thumb tip
        const dIndex = d(4, 5);  // Distance to Index Knuckle
        const dMiddle = d(4, 9); // Distance to Middle Knuckle
        const dRing = d(4, 13);  // Distance to Ring Knuckle
        const dPinky = d(4, 17); // Distance to Pinky Knuckle

        // M: "Thumb tuck in between 3 fingers" -> Thumb tip reaches Ring/Pinky side
        // It covers Index, Middle, Ring. Tip is near Ring/Pinky.
        if (dRing < T(0.3) || dPinky < T(0.35)) return "M";

        // N: "Thumb closed in 2 fingers" -> Thumb tip reaches Middle/Ring side
        // It covers Index, Middle. Tip is near Middle/Ring.
        if (dMiddle < T(0.3)) return "N";

        // T: "Thumb tuck between Index & Middle" -> Thumb tip stays near Index
        if (dIndex < T(0.35)) return "T";

        // Fallback to S if thumb is "floating" over fingers
        return "S"; 
    }

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
      wasHandPresent.current = true;
      const hand = hands[0];
      const rawGesture = recognizeGesture(hand);
      
      gestureBuffer.current.push(rawGesture);
      if (gestureBuffer.current.length > BUFFER_SIZE) gestureBuffer.current.shift();
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
        if (smoothedGesture !== "🖐️" && smoothedGesture !== "👀 Show Hand") {
          if (smoothedGesture !== lastWrittenLetter.current) {
            setText((t) => t + smoothedGesture);
            lastWrittenLetter.current = smoothedGesture;
          }
        } else {
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
      setGesture("👀 Show Hand");
      stableCount.current = 0;
      lastWrittenLetter.current = "";
      if (wasHandPresent.current) {
        setText((prev) => (prev.length > 0 && !prev.endsWith(" ") ? prev + " " : prev));
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
    <div style={{ background: "#ede7e7ff", minHeight: "100vh", color: "#210303ff", display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "flex-start", padding: "15px", boxSizing: "border-box", overflowX: "hidden" }}>
      <h1 style={{ fontSize: "clamp(0.8rem, 4vw, 1rem)", margin: "0 0 10px 0", color: "#2b9308ff" }}>
        Welcome to Signetic - Sign Language to Text Converter
      </h1>
      <h2 style={{ fontSize: "clamp(1rem, 6vw, 1.5rem)", margin: "0 0 10px 0", color: "#272704ff", minHeight: "30px" }}>
        {gesture}
      </h2>
      <div style={{ width: "100%", maxWidth: "480px", padding: "10px", border: `2px solid ${THEME_COLOR}`, borderRadius: 12, minHeight: "50px", fontSize: "1.2rem", background: `rgba(${THEME_RGB}, 0.1)`, wordWrap: "break-word", textAlign: "left", marginBottom: "15px" }}>
        {text || <span style={{ opacity: 0.5 }}>Start signing...</span>}
      </div>
      <div style={{ display: "flex", gap: "15px", marginBottom: "15px" }}>
        <button onClick={() => { setText(""); lastWrittenLetter.current = ""; }} style={{ padding: "8px 16px", background: "red", color: "#fff", border: "none", borderRadius: 8, fontSize: "0.9rem", fontWeight: "bold", cursor: "pointer" }}>CLEAR</button>
        <button onClick={() => { setText((t) => t.slice(0, -1)); lastWrittenLetter.current = ""; }} style={{ padding: "8px 16px", background: "orange", color: "#fff", border: "none", borderRadius: 8, fontSize: "0.9rem", fontWeight: "bold", cursor: "pointer" }}>BACK</button>
        <button onClick={speakText} style={{ padding: "8px 16px", background: "#2b9308ff", color: "#fff", border: "none", borderRadius: 8, fontSize: "0.9rem", fontWeight: "bold", cursor: "pointer" }}>🔊 HEAR</button>
      </div>
      <div style={{ position: "relative", width: dim.w, height: dim.h, border: `4px solid #272704ff`, borderRadius: "20px", overflow: "hidden", backgroundColor: "#000", boxShadow: `0 0 15px rgba(0,0,0,0.3)` }}>
        <Webcam ref={webcamRef} mirrored width={dim.w} height={dim.h} videoConstraints={{ facingMode: "user", aspectRatio: dim.h > dim.w ? 0.75 : 1.333 }} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
        <canvas ref={canvasRef} style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%" }} />
      </div>
    </div>
  );
}