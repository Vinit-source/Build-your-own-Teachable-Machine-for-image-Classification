# Complete In-Repo Reference (Pages #0 to #16)

This document is a self-contained learning reference for the full activity flow. Students can complete the project using only this repository.

## How To Use This Reference

1. Keep this document open while coding.
2. Follow pages in order from #0 to #16.
3. Use the checkpoints at the end of each page before moving on.
4. If a page says Guided Build, write the code yourself from hints.
5. If a page says Scaffolded Build, fill in blanks in script.js.

## Project Files You Will Use

- index.html
- style.css
- script.js
- docs/checkpoints.md
- docs/teacher-key.md (for instructor use)

## Page #0: Before You Begin

### Purpose
Understand what transfer learning is and why it is practical in the browser.

### Core Ideas
- Transfer learning reuses a model that already learned useful image features.
- You train only a small new classifier head for your custom classes.
- This is faster and needs fewer examples than full training from scratch.

### Checkpoint
- You can explain why reusing a base model saves time and data.

---

## Page #1: What Is TensorFlow.js?

### Purpose
Understand what TensorFlow.js does and where it runs.

### Core Ideas
- TensorFlow.js runs machine learning in JavaScript environments.
- Browser execution can improve privacy and reduce server dependence.
- Available backends include GPU (WebGL), WASM, and CPU fallback.

### Checkpoint
- You can name one benefit of browser ML and one fallback backend.

---

## Page #2: Transfer Learning

### Purpose
Understand how feature extractors and classifier heads work together.

### Core Ideas
- A pre-trained network has rich low-level and mid-level feature detectors.
- You can remove the original final classifier and attach a new head.
- In this workflow, the base is fixed and only the head is trained.

### Checkpoint
- You can describe frozen base model plus trainable head in one paragraph.

---

## Page #3: TensorFlow Hub Base Models

### Purpose
Choose an appropriate reusable model for image features.

### Core Ideas
- Look for Image Feature Vector models when doing transfer learning.
- For this project, the model is loaded as a graph model.
- Graph model loading uses the TensorFlow.js graph-model API.

### Checkpoint
- You know why feature-vector models are preferred over ready-made label heads.

---

## Page #4: Get Set Up To Code

### Purpose
Prepare local execution environment.

### Actions
1. Run a local server from project root:
   python3 -m http.server 8000
2. Open http://localhost:8000
3. Open browser console.

### Important
Do not open the project through file:// because browser permissions and module loading can fail.

### Checkpoint
- Page loads with no startup errors in console.

---

## Page #5: App HTML Boilerplate

### Purpose
Understand the UI structure required for data collection and training.

### Key Elements
- Status area for model and prediction updates.
- Video element for webcam stream.
- Enable webcam button.
- Data collector buttons with numeric class ids and class names.
- Train and reset buttons.

### Why data attributes matter
- data-1hot stores numeric class id starting at 0.
- data-name stores human-readable class label.

### Checkpoint
- You can identify all controls in index.html and explain each one.

---

## Page #6: Add Style

### Purpose
Make the interface readable and functional for capture and prediction.

### Key Rules
- Webcam area must be visible.
- Buttons must be obvious and clickable.
- Status text should be easy to read.
- Hidden state class should remove the enable button after webcam starts.

### Checkpoint
- UI is usable on your screen and webcam panel is visible.

---

## Page #7: JavaScript Key Constants And Listeners

### Purpose
Set global constants, mutable app state, and event routing.

### In script.js this section controls
- DOM references for status, video, and buttons.
- Model input dimensions for MobileNet.
- Sentinel stop state for data capture.
- Dynamic class-name population from HTML buttons.
- Event wiring for webcam, train, reset, and data collection.

### Important behavior note
Data collection uses mousedown and mouseup. Releasing outside button can miss mouseup and keep collection active.

### Checkpoint
- You can trace how a button press reaches its handler.

---

## Page #8: Load The MobileNet Base Model (Guided Build)

### Purpose
Load and warm up the reusable feature extractor.

### Student Task (write code in script.js)
Complete loadMobileNetFeatureModel so it:
1. Defines the TFHub MobileNet feature-vector URL.
2. Loads using graph-model API with fromTFHub option.
3. Stores model in mobilenet.
4. Updates status text when loaded.
5. Performs one warm-up forward pass with zeros tensor sized 1 x 224 x 224 x 3.
6. Logs warm-up output shape.
7. Is called at startup.

### Expected output
- Status confirms model loaded.
- Console shows output shape 1 x 1024.

### Why warm-up is used
First call on larger models can be slower due to internal preparation. Warm-up pays that cost early.

### Checkpoint
- loadMobileNetFeatureModel is fully implemented and output shape is correct.

---

## Page #9: Define The New Model Head (Scaffolded Build)

### Purpose
Create a small trainable classifier on top of fixed feature vectors.

### Student Task (fill blanks in script.js)
Set values for:
- input feature length
- hidden units
- hidden activation
- output activation

### Architecture intent
- Dense hidden layer receives MobileNet feature vector.
- Dense output layer size equals number of classes.
- summary call prints architecture for verification.
- compile uses adaptive optimizer and class-appropriate loss.

### Typical values used in this activity
- input feature size: 1024
- hidden units: 128 (or nearby)
- hidden activation: relu
- output activation: softmax

### Checkpoint
- summary output is visible and output units match class count.

---

## Page #10: Enable The Webcam

### Purpose
Start webcam stream and mark data source as active.

### What code does
- Checks browser camera API availability.
- Requests webcam stream with practical width and height.
- Sets video element source object to stream.
- Waits for loadeddata event to mark videoPlaying true.
- Hides enable button after successful stream start.

### Checkpoint
- Webcam feed appears and enable button is hidden.

---

## Page #11: Data Collection Button Event Handler

### Purpose
Toggle current collection class while button is held.

### What code does
- Reads numeric class id from button attribute.
- If currently idle, starts collection for that class.
- If already collecting, stops collection.
- Starts capture loop call.

### Checkpoint
- Holding and releasing a class button starts and stops collection behavior.

---

## Page #12: Data Collection Loop

### Purpose
Capture frames, convert to features, and store training examples.

### Processing pipeline each frame
1. Read frame from video.
2. Resize to model input size.
3. Normalize pixel range.
4. Expand dims for batch input.
5. Run base model to obtain feature vector.
6. Squeeze batch dimension.
7. Save feature vector and current class id.
8. Increment class counters.
9. Update status text with counts.
10. Request next frame while collection remains active.

### Memory practice
Intermediate tensors are created inside tidy scope so they are released automatically.

### Checkpoint
- Counts increase while button is held and stop when released.

---

## Page #13: Train And Predict

### Purpose
Train the classifier head on collected features and run live predictions.

### Training stage sequence
1. Pause prediction loop.
2. Shuffle feature-label pairs together.
3. Convert labels to integer tensor.
4. Convert integers to one-hot targets.
5. Stack feature tensors into one training tensor.
6. Fit model for multiple epochs with mini-batches.
7. Dispose temporary tensors.
8. Resume prediction loop.

### One-hot outputs: what and why
- Class labels start as integers (for example 0, 1, 2).
- One-hot encoding converts each label into a vector with one 1 and remaining 0s.
- For 3 classes:
  - class 0 -> [1, 0, 0]
  - class 1 -> [0, 1, 0]
  - class 2 -> [0, 0, 1]
- The model output is a probability distribution across classes.
- Cross-entropy compares output probabilities to one-hot targets.
- This teaches the model to push confidence toward the correct class and reduce others.

### Prediction stage sequence
1. Read and normalize current frame.
2. Resize and extract features through base model.
3. Run classifier head.
4. Take argmax class index.
5. Display class name and confidence percent.
6. Request next animation frame.

### Checkpoint
- Training logs appear.
- Live predictions update with class and confidence.
- You can explain one-hot targets clearly.

---

## Page #14: Implement Reset

### Purpose
Clear collected data safely so users can retrain.

### What code must do
- Stop predictions.
- Reset state flags and counters.
- Dispose all stored feature tensors first.
- Then clear input and output arrays.
- Update status.
- Optionally print tensor count for sanity.

### Why dispose order matters
If array references are dropped before tensor disposal, tensors can remain allocated.

### Checkpoint
- After reset, predictions stop and counts are cleared.

---

## Page #15: Let Us Try It Out

### End-to-end test
1. Enable webcam.
2. Gather around 30 examples per class.
3. Train model.
4. Show objects to camera and verify class switching.
5. Reset and repeat with improved data.

### What to observe
- Better class separation with more and more varied examples.
- Faster retraining than full model training.

### Checkpoint
- Full pipeline works without reloading the page.

---

## Page #16: Congratulations And Next Steps

### Recap outcomes
You have now:
- applied transfer learning in-browser,
- loaded a reusable feature extractor,
- trained a custom head from webcam data,
- produced real-time classification outputs.

### Suggested extensions
- Add a third class button and retrain.
- Tune hidden units and compare confidence stability.
- Improve capture quality with more varied lighting and angles.
- Replace stretch-resize with centered crop as an advanced improvement.

### Checkpoint
- You can describe at least one extension and why it may improve results.

---

## Troubleshooting Quick Reference

### Webcam does not start
- Recheck browser permission.
- Use localhost server, not file path.

### Model never trains
- Confirm Step #8 and Step #9 are complete.
- Confirm each class has examples.

### Prediction quality is poor
- Increase examples per class.
- Ensure classes are visually distinct.
- Collect examples across different positions and lighting.

### Memory keeps growing
- Confirm tidy usage in loops.
- Confirm reset disposes stored tensors before clearing arrays.

---

## Assessment Prompts For Learners

1. Explain transfer learning in your own words.
2. Explain why one-hot targets are required for multi-class training.
3. Explain why model warm-up is done once after loading.
4. Explain why resize and normalization happen before feature extraction.
5. Explain why reset disposes tensors before array clear.
