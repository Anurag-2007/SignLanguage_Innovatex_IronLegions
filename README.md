

# Signetic 

**Real-time ML Sign Language to Text Converter**



**Signetic** is an accessibility-focused web application that translates American Sign Language (ASL) fingerspelling into text in real-time. Built with TensorFlow.js and Computer Vision, it runs entirely in the browser, offering a privacy-first, low-latency solution for bridging communication gaps.

---

## 🚀 Tech Stack

* **Frontend Library:** [React.js](https://reactjs.org/) ( Functional Components)
* **Machine Learning:** [TensorFlow.js](https://www.tensorflow.org/js) (Client-side ML)
* **Computer Vision:** [MediaPipe Hands](https://www.google.com/search?q=https://google.github.io/mediapipe/solutions/hands) (Hand landmark detection)
* **State Management:** React `useState`, `useRef`, `useCallback`
* **Web speech API:** Web Speech API (Text-to-Speech)
* **Styling:** CSS-in-JS (Responsive Design)

---

## 💡 Use Case

The primary goal of **Signetic** is to facilitate communication for the Deaf and Hard-of-Hearing community in digital environments.

1. **Educational Tool:** Helps learners practice ASL fingerspelling with instant feedback.
2. **Communication Bridge:** Allows non-signers to understand fingerspelling without an interpreter.
3. **Accessibility Interface:** Can be integrated into kiosks or video calls to provide a silent text-input method.

---

## ✨ Key Features

### 1. Real-Time Detection

Uses a highly optimized TensorFlow model to detect 26 alphabets (A-Z) and specialized hand shapes instantly via the webcam.

### 2. Smart Stabilization

Includes a **smoothing algorithm** (Buffer & Threshold) to prevent "flickering" text. It waits for the user to hold a gesture steady for a few frames before typing it.

### 3. Auto-Space Functionality

Typing sentences is natural. Simply **move your hand out of the camera frame**, and the app automatically adds a `Space`, allowing for fluid sentence construction.

### 4. Accessibility Tools

* **🔊 Hear Button:** Uses the Web Speech API (TTS) to read the typed text aloud, enabling two-way communication.
* **⌫ Backspace:** Supports both the on-screen button and physical keyboard `Backspace` key to correct mistakes.
* **⌨️ Keyboard Support:** Use `Escape` to clear text and `Spacebar` for manual spacing.

### 5. Responsive Design

* **Desktop:** Compact video feed to allow multitasking.
* **Mobile:** Auto-detects portrait orientation and adjusts the camera aspect ratio (3:4) for a native app feel.

---

## 🛠️ Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/your-username/signetic.git
cd signetic

```


2. **Install dependencies:**
```bash
npm install
# or
yarn install

```


3. **Run the application:**
```bash
npm start

```


Open [https://signlang-ironlegions.vercel.app](https://signlang-ironlegions.vercel.app) to view it in your browser.

---

## ✋ Supported Signs Demo

Signetic supports the full ASL alphabet with specialized logic for tricky gestures.

### **Static Letters**

| Letter | Description |
| --- | --- |
| **A, E, S** | **Fist Variations:** Distinguishes based on thumb position (Side vs. Curled vs. Crossing fingers). |
| **B, C, D** | **Open Hand Variations:** Tracks thumb tucks and index extension. |
| **I, Y** | **Pinky Variations:** Distinguishes "I" (Fist + Pinky) vs "Y" (Thumb + Pinky). |

### **Complex/Orientation Letters**

| Gesture | Logic Used |
| --- | --- |
| **K** | ✌️ **Upward V** with thumb tucked inside. |
| **P** | 👇 **Downward V** with thumb tucked inside (Inverse of K). |
| **L** | 👆 Index up, Thumb out (90° angle). |
| **Q** | 👇 **Downward L** (Index & Thumb pointing down). |
| **M** | ✊ Thumb tucked between Ring & Pinky finger. |
| **N** | ✊ Thumb tucked between Middle & Ring finger. |
| **T** | ✊ Thumb tucked between Index & Middle finger. |
| **R** | 🤞 Crossed Index and Middle fingers. |
| **X** | 🪝 Hooked Index finger. |

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for improving the detection accuracy of dynamic signs (like J and Z), feel free to fork the repo and submit a PR.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewSign`)
3. Commit your Changes (`git commit -m 'Add support for J'`)
4. Push to the Branch (`git push origin feature/NewSign`)
5. Open a Pull Request

