/*
  COMPLETE IN-CODE REFERENCE (Pages #0 to #16)

  Pages #0-#4 (concept + setup):
  - #0: Transfer learning = reuse a trained base model + train a small new head.
  - #1: TensorFlow.js runs in JavaScript environments (browser, Node.js, etc.).
  - #2: Freeze feature extractor, train only classifier head for speed/data efficiency.
  - #3: Choose feature-vector model from TFHub for transfer learning.
  - #4: Run via local server (not file://) and use browser console for checks.

  Pages #5-#6 (HTML/CSS):
  - HTML must provide video, status, class buttons, train and reset actions.
  - CSS must keep webcam and controls clearly visible and usable.

  Pages #7-#14 (this file):
  - #7: constants, state, and listeners.
  - #8: load + warm up MobileNet.
  - #9: build the classifier head.
  - #10: enable webcam.
  - #11: handle class-button press/release.
  - #12: collect feature vectors and labels.
  - #13: train with one-hot targets and run prediction loop.
  - #14: reset data and dispose tensors safely.

  Pages #15-#16 (test + extension):
  - #15: gather around 30 samples/class, train, verify live predictions.
  - #16: extend by adding classes, tuning head size, or improving data quality.
*/

// ----- Page #7: DOM references and constants used across the full app pipeline -----
// Checkpoint for this block:
// - You can point to where each UI element is referenced and why.
// - You can explain why the input size is fixed to 224x224 for this model.
// STATUS: human-readable runtime feedback for students.
const STATUS = document.getElementById('status');
// VIDEO: camera stream target; frames from this element become tensors later.
const VIDEO = document.getElementById('webcam');
const ENABLE_CAM_BUTTON = document.getElementById('enableCam');
const RESET_BUTTON = document.getElementById('reset');
const TRAIN_BUTTON = document.getElementById('train');

// MobileNet v3 feature-vector model expects 224x224 RGB images.
const MOBILE_NET_INPUT_WIDTH = 224;
const MOBILE_NET_INPUT_HEIGHT = 224;

// Sentinel for "not collecting class data right now".
const STOP_DATA_GATHER = -1;

// Populated from button data-name attributes; used to display class labels.
const CLASS_NAMES = [];

// ----- Page #7: shared mutable state -----
let mobilenet = undefined; // Will hold loaded TFHub graph model.
let model = undefined; // Will hold trainable classifier head.
let gatherDataState = STOP_DATA_GATHER; // Active class id while button is held.
let videoPlaying = false; // True once webcam stream has loaded.
let trainingDataInputs = []; // Array of feature tensors (each expected length 1024).
let trainingDataOutputs = []; // Array of numeric class ids.
let examplesCount = []; // Per-class count for UI feedback.
let predict = false; // Controls prediction animation loop.

// ----- Page #7: primary button listeners -----
ENABLE_CAM_BUTTON.addEventListener('click', enableCam);
TRAIN_BUTTON.addEventListener('click', trainAndPredict);
RESET_BUTTON.addEventListener('click', reset);

// ----- Page #7: dynamic class button wiring -----
// mousedown starts data collection; mouseup stops it.
// This mirrors the codelab hold-to-record behavior.
const dataCollectorButtons = document.querySelectorAll('button.dataCollector');
for (let i = 0; i < dataCollectorButtons.length; i++) {
  dataCollectorButtons[i].addEventListener('mousedown', gatherDataForClass);
  dataCollectorButtons[i].addEventListener('mouseup', gatherDataForClass);

  // Human-friendly class names come from HTML so class count can scale without JS edits.
  CLASS_NAMES.push(dataCollectorButtons[i].getAttribute('data-name'));
}

// -------------------------------
// STEP 8 (Page #8: Load base model)
// -------------------------------
// Page #8 objective: load MobileNet feature-vector model from TFHub, then warm it up.
// Keep this student-authored. Do not paste source codelab code directly.
//
// Required behavior for this function:
// 1) Define a TFHub feature-vector URL for MobileNet v3 small 224.
// 2) Call tf.loadGraphModel(url, { fromTFHub: true }) and store in "mobilenet".
// 3) Update STATUS to show successful load.
// 4) Run one warm-up pass inside tf.tidy with tf.zeros([1, 224, 224, 3]).
// 5) Log output shape and verify it is [1, 1024].
// 6) Keep function async and call it once at startup.
// Success criteria:
// - STATUS indicates successful base-model load.
// - Console includes [1, 1024] after warm-up call.
async function loadMobileNetFeatureModel() {
  STATUS.innerText = 'Step 8 pending: implement MobileNet load and warmup in script.js.';
}

// -------------------------------
// STEP 9 (Page #9: Define model head)
// -------------------------------
// Page #9 objective: define the small trainable head on top of MobileNet features.
// Students fill these blanks and verify model.summary() output.
// Key expectation from this page:
// - Input layer consumes MobileNet features.
// - Output layer uses number of classes in UI.
// - summary() is used to inspect shape/parameter correctness.

// BLANK #1: should match feature count from MobileNet warm-up output.
const STEP9_INPUT_FEATURES = null;
// BLANK #2: hidden layer width (tradeoff between capacity and speed).
const STEP9_HIDDEN_UNITS = null;
// BLANK #3: common hidden activation for this setup.
const STEP9_HIDDEN_ACTIVATION = null;
// BLANK #4: output activation for class probabilities.
const STEP9_OUTPUT_ACTIVATION = null;

function buildModelHead() {
  // Guardrail: training should not run until all Step 9 blanks are complete.
  if (
    !Number.isFinite(STEP9_INPUT_FEATURES) ||
    !Number.isFinite(STEP9_HIDDEN_UNITS) ||
    !STEP9_HIDDEN_ACTIVATION ||
    !STEP9_OUTPUT_ACTIVATION
  ) {
    STATUS.innerText =
      'Step 9 pending: fill the model head blanks in script.js before training.';
    return undefined;
  }

  const localModel = tf.sequential();

  // First dense layer consumes feature vectors produced by MobileNet.
  localModel.add(
    tf.layers.dense({
      inputShape: [STEP9_INPUT_FEATURES],
      units: STEP9_HIDDEN_UNITS,
      activation: STEP9_HIDDEN_ACTIVATION,
    })
  );

  // Output layer dimension equals current number of classes.
  localModel.add(
    tf.layers.dense({
      units: CLASS_NAMES.length,
      activation: STEP9_OUTPUT_ACTIVATION,
    })
  );

  // Required page #9 checkpoint: inspect shapes and parameter counts.
  // Expected pattern: (null, 1024) -> hidden -> (null, CLASS_NAMES.length).
  localModel.summary();

  // Binary vs categorical loss follows class-count rule from codelab explanation.
  localModel.compile({
    optimizer: 'adam',
    loss:
      CLASS_NAMES.length === 2 ? 'binaryCrossentropy' : 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return localModel;
}

// ----- Page #10: webcam capability check -----
// If this fails, verify secure context (localhost/https) and browser permission.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// ----- Page #10: enable webcam and mark stream as ready -----
// Page #10 checkpoint:
// - Video stream visible.
// - Enable button hidden after stream load.
function enableCam() {
  if (!hasGetUserMedia()) {
    console.warn('getUserMedia() is not supported by your browser');
    return;
  }

  // Codelab uses 640x480 to avoid unnecessary high-res capture cost before 224x224 resize.
  const constraints = {
    video: true,
    width: 640,
    height: 480,
  };

  navigator.mediaDevices
    .getUserMedia(constraints)
    .then((stream) => {
      VIDEO.srcObject = stream;
      VIDEO.addEventListener('loadeddata', () => {
        videoPlaying = true;
        ENABLE_CAM_BUTTON.classList.add('removed');
        STATUS.innerText = 'Webcam ready. Gather data for each class.';
      });
    })
    .catch((error) => {
      console.error(error);
      STATUS.innerText = 'Could not access webcam. Check browser permissions.';
    });
}

// ----- Page #11: class button event handler -----
// Page #11 checkpoint:
// - Holding button starts data capture; releasing stops capture.
function gatherDataForClass() {
  if (!mobilenet) {
    STATUS.innerText = 'Load MobileNet first (complete Step 8).';
    return;
  }

  // data-1hot is a string attribute; parse to integer class id.
  const classNumber = parseInt(this.getAttribute('data-1hot'), 10);

  // Toggle start/stop collection: mousedown sets class id, mouseup returns to sentinel.
  gatherDataState =
    gatherDataState === STOP_DATA_GATHER ? classNumber : STOP_DATA_GATHER;

  // Start or continue frame sampling loop.
  dataGatherLoop();
}

// ----- Page #12: collect feature vectors while class button is held -----
// Page #12 key ideas:
// - fromPixels gives [0..255] values, so normalize to [0..1].
// - resize to 224x224 before model inference.
// - store extracted features (not raw image pixels) for training.
// - update class counts every sampled frame.
function dataGatherLoop() {
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    // tf.tidy disposes intermediate tensors; only returned feature tensor is kept.
    const imageFeatures = tf.tidy(() => {
      // 1) Read current webcam frame.
      const videoFrameAsTensor = tf.browser.fromPixels(VIDEO);

      // 2) Resize to MobileNet input size.
      // Note: this simple resize stretches 640x480 to square, acceptable for this exercise.
      const resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
        true
      );

      // 3) Normalize [0,255] pixels into [0,1].
      const normalizedTensorFrame = resizedTensorFrame.div(255);

      // 4) Expand to batch=1, run feature extraction, squeeze back to 1D.
      return mobilenet.predict(normalizedTensorFrame.expandDims()).squeeze();
    });

    // Store features plus numeric class label.
    trainingDataInputs.push(imageFeatures);
    trainingDataOutputs.push(gatherDataState);

    // Track per-class sample counts for live feedback.
    if (examplesCount[gatherDataState] === undefined) {
      examplesCount[gatherDataState] = 0;
    }
    examplesCount[gatherDataState]++;

    STATUS.innerText = '';
    for (let n = 0; n < CLASS_NAMES.length; n++) {
      STATUS.innerText += `${CLASS_NAMES[n]} data count: ${examplesCount[n] || 0}. `;
    }

    // Request next frame while still gathering data.
    window.requestAnimationFrame(dataGatherLoop);
  }
}

// ----- Page #13: train the head model and start prediction loop -----
// Page #13 checkpoints:
// - model.fit logs epochs in console.
// - prediction loop resumes after training.
// - status shows class + confidence.
async function trainAndPredict() {
  if (!mobilenet) {
    STATUS.innerText = 'Step 8 incomplete: load MobileNet first.';
    return;
  }

  if (trainingDataInputs.length === 0) {
    STATUS.innerText = 'Collect data before training.';
    return;
  }

  // Require at least one sample per class to avoid incomplete label space.
  for (let i = 0; i < CLASS_NAMES.length; i++) {
    if (!examplesCount[i]) {
      STATUS.innerText = `Collect data for ${CLASS_NAMES[i]} before training.`;
      return;
    }
  }

  // Build once; reuse on later retrains unless page reloads.
  if (!model) {
    model = buildModelHead();
  }

  if (!model) {
    return;
  }

  // Pause prediction during training to avoid overlapping loops.
  predict = false;

  // Shuffle paired arrays to reduce order bias.
  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);

  // One-hot explanation (page #13 emphasis):
  // - Labels start as integers: 0, 1, 2, ...
  // - oneHot converts each label to a class-length target vector.
  //   Example for 3 classes: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1].
  // - This lets cross-entropy compare predicted class probabilities
  //   against explicit target distributions.
  const outputsAsTensor = tf.tensor1d(trainingDataOutputs, 'int32');
  const oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);

  // Stack array of feature tensors into [numExamples, featureSize].
  // For this project featureSize is expected to be 1024.
  const inputsAsTensor = tf.stack(trainingDataInputs);

  STATUS.innerText = 'Training... watch console for epoch logs.';
  await model.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: logProgress },
  });

  // Dispose temporary tensors created for fit input/output.
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();

  predict = true;
  STATUS.innerText = 'Training complete. Running live predictions.';
  predictLoop();
}

// Epoch-by-epoch metrics are useful when teaching loss/accuracy trends.
function logProgress(epoch, logs) {
  console.log(`Epoch ${epoch}`, logs);
}

// ----- Page #13: core prediction loop -----
// Runs continuously using requestAnimationFrame while "predict" stays true.
function predictLoop() {
  if (predict) {
    tf.tidy(() => {
      // Same preprocessing used in data collection keeps train/inference consistent.
      const videoFrameAsTensor = tf.browser.fromPixels(VIDEO).div(255);
      const resizedTensorFrame = tf.image.resizeBilinear(
        videoFrameAsTensor,
        [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
        true
      );

      // Extract features, run classifier head, and flatten output probabilities.
      const imageFeatures = mobilenet.predict(resizedTensorFrame.expandDims());
      const prediction = model.predict(imageFeatures).squeeze();

      // Highest-probability index is predicted class.
      const highestIndex = prediction.argMax().arraySync();
      const predictionArray = prediction.arraySync();

      STATUS.innerText =
        `Prediction: ${CLASS_NAMES[highestIndex]} with ` +
        `${Math.floor(predictionArray[highestIndex] * 100)}% confidence`;
    });

    window.requestAnimationFrame(predictLoop);
  }
}

// ----- Page #14: reset collected data while keeping loaded models in memory -----
// Page #14 key memory rule:
// - Dispose tensors first, then clear arrays.
// - Base model and classifier head stay loaded for quick retraining.
function reset() {
  predict = false;
  gatherDataState = STOP_DATA_GATHER;
  examplesCount.length = 0;

  // Important memory note from codelab: dispose tensors before clearing arrays.
  for (let i = 0; i < trainingDataInputs.length; i++) {
    trainingDataInputs[i].dispose();
  }
  trainingDataInputs.length = 0;
  trainingDataOutputs.length = 0;

  STATUS.innerText = 'No data collected.';
  console.log('Tensors in memory:', tf.memory().numTensors);
}

// Pages #15-#16 practical run checklist:
// 1) Enable webcam.
// 2) Collect meaningful samples for each class (often around 30 per class works well).
// 3) Train and verify prediction changes as objects change.
// 4) Reset and retrain with better data when needed.
// 5) Extend by adding classes or adjusting hidden units.

// Startup path: page #8 work should eventually make this load the base model immediately.
loadMobileNetFeatureModel();
