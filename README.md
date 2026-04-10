# Teachable Machine Clone for Image Classification

This repository is a student template to replicate the TensorFlow.js transfer learning codelab flow from page #0 through page #16.

## Goal
Students build a browser app that:
- Loads a pre-trained MobileNet feature extractor
- Collects webcam examples for each class
- Trains a small custom model head
- Predicts classes live from webcam frames

## What Is Included
- `index.html`: Classroom UI scaffold
- `style.css`: Layout and visibility styles
- `script.js`: Starter pipeline with guided gaps for key learning steps
- `docs/student-template.md`: Step-by-step worksheet mapped to pages #0-#16
- `docs/checkpoints.md`: Verification rubric and troubleshooting
- `docs/teacher-key.md`: Instructor reference for answers and grading cues

## Source Of Truth
- Use `script.js` comments as the primary implementation reference.
- `index.html` and `style.css` are aligned to the same page-by-page numbering style.
- Docs mirror the code flow, but if wording differs, follow `script.js`.

## Local Run
1. Open a terminal in this folder.
2. Run a local server:
   - `python3 -m http.server 8000`
3. Open `http://localhost:8000` in a modern browser.

Do not open the app using `file://`.

## Student Focus Steps
- Step 8: Guidance-only implementation for loading and warming MobileNet
- Step 9: Fill the Step 9 constants in `script.js` and inspect `model.summary()` in console
- Step 13: Explain one-hot outputs and why they are needed for training

## Attribution
This classroom template is inspired by and aligned to the TensorFlow.js codelab sequence:
https://codelabs.developers.google.com/tensorflowjs-transfer-learning-teachable-machine
