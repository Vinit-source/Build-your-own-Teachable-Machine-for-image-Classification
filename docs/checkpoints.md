# Checkpoints and Troubleshooting (Pages #0 to #16)

## Required Checkpoints
- Page #0: Explain transfer learning and one advantage.
- Page #4: App runs from local server on localhost.
- Page #8: MobileNet loads and warm-up output shape is `[1, 1024]`.
- Page #9: `model.summary()` shows expected layers and dimensions.
- Page #12: Data counts increase while holding class buttons.
- Page #13: Training logs epochs and prediction status updates live.
- Page #14: Reset clears collected examples and stops prediction loop.
- Page #15: End-to-end workflow succeeds in one full run.

## Common Failure Modes
1. Camera does not open.
- Cause: permissions denied or insecure context.
- Fix: use localhost or https and re-allow camera in browser settings.

2. Training button does nothing.
- Cause: Step 8 or Step 9 not completed.
- Fix: complete MobileNet load and model-head blanks in `script.js`.

3. Shape mismatch during training.
- Cause: incorrect feature input size for model head.
- Fix: verify MobileNet warm-up output and set head input shape accordingly.

4. Predictions are unstable.
- Cause: too little or imbalanced data.
- Fix: collect more examples for each class and retrain.

5. Memory grows after repeated experiments.
- Cause: tensors not disposed during reset or loops.
- Fix: verify `tf.tidy` usage and dispose loops in reset logic.

## Teacher Review Rubric (Quick)
- Accuracy: student completes Step 8 and Step 9 without direct copy from source page.
- Understanding: student one-hot explanation in Step 13 is clear and technically correct.
- Execution: student can train, predict, reset, and retrain successfully.
- Debugging: student can identify one likely issue and a valid fix path.
