# Student Template: Page-by-Page Guide (#0 to #16)

This worksheet follows the same progression as the source codelab pages.

Use `script.js` as the implementation source-of-truth. This file explains the same flow.

## Page #0: Before You Begin
- Objective: understand what transfer learning is and why it is practical in the browser.
- Do: explain in your own words why we reuse a base model instead of training from scratch.
- Check: you can name two transfer-learning benefits.

## Page #1: What Is TensorFlow.js?
- Objective: understand where TensorFlow.js runs.
- Do: list one client-side and one server-side runtime option.
- Check: you can explain one privacy and one speed advantage.

## Page #2: Transfer Learning
- Objective: understand frozen feature extractor plus trainable head.
- Do: draw a simple diagram: input image -> base model features -> classifier head.
- Check: you can explain what is frozen and what is trainable.

## Page #3: TensorFlow Hub Base Models
- Objective: understand how to choose feature-vector models.
- Do: inspect TFHub and identify a MobileNet feature-vector model.
- Check: confirm model type is compatible with `tf.loadGraphModel`.

## Page #4: Get Set Up to Code
- Objective: run the project from a local server.
- Do: start `python3 -m http.server 8000` or use Live Server extension.
- Check: app opens on `http://localhost:8000` (or `http://localhost:5173`) with no startup errors.

## Page #5: App HTML Boilerplate
- Objective: verify UI structure.
- Do: inspect buttons and ensure each class button has `data-1hot` and `data-name`.
- Check: status text appears and webcam container is visible.

## Page #6: Add Style
- Objective: keep controls visible and readable.
- Do: confirm button layout and responsive video sizing.
- Check: UI remains usable on both laptop and smaller screens.

## Page #7: Key Constants and Listeners
- Objective: understand app state variables and events.
- Do: identify where class names are populated and why mousedown/mouseup are used.
- Check: no console errors after load.

## Page #8: Load the MobileNet Base Model (Guidance Only)
Important: implement this yourself. Do not copy the page code directly.

- Objective: load feature extractor and warm it up.
- Do:
  1. Define a TFHub MobileNet feature-vector URL.
  2. Load with `await tf.loadGraphModel(url, { fromTFHub: true })`.
  3. Update status text when loaded.
  4. Warm up the model with a zero tensor for one forward pass.
  5. Log output shape to console.
- Hints:
  - Input shape should use `MOBILE_NET_INPUT_HEIGHT`, `MOBILE_NET_INPUT_WIDTH`, and 3 channels.
  - Warm-up should run inside `tf.tidy`.
- Check:
  - status says model loaded.
  - console shows output shape `[1, 1024]`.

## Page #9: Define the New Model Head (Scaffolded Fill Blanks)
Fill the Step 9 constants in `script.js` where Step 9 markers are located.

```javascript
const STEP9_INPUT_FEATURES = _____;
const STEP9_HIDDEN_UNITS = _____;
const STEP9_HIDDEN_ACTIVATION = '_____';
const STEP9_OUTPUT_ACTIVATION = '_____';
```

- Objective: create a small trainable head on top of MobileNet features.
- Do: choose valid values for input feature count, hidden units, and activations.
- Check:
  - run and inspect `model.summary()` output.
  - verify output units match number of classes.

## Page #10: Enable Webcam
- Objective: stream webcam into the app.
- Do: allow camera permission and verify button hides after stream starts.
- Check: live video appears in the page.

## Page #11: Data Collection Button Handler
- Objective: toggle collection state per class.
- Do: hold and release class buttons.
- Check: state switches between class id and stop sentinel.

## Page #12: Data Collection Loop
- Objective: collect feature vectors plus labels.
- Do: capture at least 10-30 examples per class.
- Check: status text counts increase while holding a class button.

## Page #13: Train and Predict (One-Hot Focus)
- Objective: train the model head and run live inference.
- Do: run training and observe epoch logs.
- Check: predictions update with confidence percentages.

One-hot outputs explained:
- Your labels may start as class indexes like `[0, 1, 0, 1]`.
- One-hot encoding turns each label into a target vector.
- Example with 3 classes:
  - class 0 -> `[1, 0, 0]`
  - class 1 -> `[0, 1, 0]`
  - class 2 -> `[0, 0, 1]`
- Why this helps:
  - the model outputs class probabilities (softmax), one probability per class.
  - training compares those probabilities to one-hot targets using cross-entropy loss.
  - this teaches the model to put high probability on the correct class and low probability on others.

Student checkpoint writing task:
- Write 3-5 sentences describing what one-hot targets are and what they achieve in this training loop.

## Page #14: Implement Reset
- Objective: clear collected data safely.
- Do: reset and verify tensors were disposed before arrays were cleared.
- Check: status resets and predictions stop.

## Page #15: Try It Out
- Objective: run full workflow end-to-end.
- Do:
  1. Enable webcam
  2. Gather class data
  3. Train
  4. Test live prediction
  5. Reset and retrain
- Check: model adapts after collecting better data.

## Page #16: Congratulations and Extensions
- Objective: reflect and extend.
- Do one extension:
  - add a third class button,
  - change hidden layer units,
  - discuss overfitting with tiny datasets.
- Check: explain how your extension affected behavior.
