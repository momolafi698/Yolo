import { InferenceSession, Tensor } from 'onnxruntime-node';
import path from 'path';

async function check() {
  const modelDir = 'c:\\Users\\momol\\OneDrive - SAS L\'Ecole LDLC\\Bureau\\yolo\\Yolo\\public\\models';
  
  for (const modelFile of ['yolo26n-pose.onnx', 'yolo26s-pose.onnx']) {
    const fullPath = path.join(modelDir, modelFile);
    try {
      console.log(`\n=== Loading ${modelFile} ===`);
      const session = await InferenceSession.create(fullPath);
      
      const dummyInput = new Tensor('float32', new Float32Array(1 * 3 * 640 * 640), [1, 3, 640, 640]);
      const results = await session.run({ images: dummyInput });
      
      console.log('Output keys:', Object.keys(results));
      for (const key of Object.keys(results)) {
        console.log(` - ${key} shape:`, results[key].dims);
      }
    } catch (err) {
      console.error(`Error loading ${modelFile}:`, err);
    }
  }
}

check();
