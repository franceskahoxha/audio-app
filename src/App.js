import React, { useState } from "react";
import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile } from "@ffmpeg/util";
import * as ort from "onnxruntime-web";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  const ffmpeg = new FFmpeg();

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    if (file) setSelectedFile(file);
  };

  const alertError = (message, error) => {
    console.error(message, error);
    alert(message);
  };

  const convertAudio = async (file) => {
    const inputFileName = "input.mp3";
    const outputFileName = "output_24khz.wav";

    console.log("Loading FFmpeg...");
    await ffmpeg.load();
    await ffmpeg.writeFile(inputFileName, await fetchFile(file));
    await ffmpeg.exec([
      "-i",
      inputFileName,
      "-ar",
      "24000",
      "-ac",
      "1",
      "-sample_fmt",
      "s16",
      outputFileName,
    ]);

    console.log("Audio successfully converted to 24kHz.");
    return await ffmpeg.readFile(outputFileName);
  };

  const prepareInferenceSession = async () => {
    const session = await ort.InferenceSession.create("/encodec_model.onnx");
    return session;
  };

  const runInference = async (session, audioBuffer) => {
    const tensor = new ort.Tensor("float32", audioBuffer, [1, 1, audioBuffer.length]);
    const feeds = { [session.inputNames[0]]: tensor };
    return await session.run(feeds);
  };

  const convertBigIntToNumber = (bigIntArray) =>
    Array.from(bigIntArray).map((value) => {
      const numValue = Number(value);
      if (
        numValue > Number.MAX_SAFE_INTEGER ||
        numValue < Number.MIN_SAFE_INTEGER
      ) {
        throw new Error("BigInt value exceeds safe range for JSON serialization.");
      }
      return numValue;
    });

  const sendToBackend = async (encodedData, audioScales) => {
    const response = await fetch("http://127.0.0.1:8000/encode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        encoded_data: convertBigIntToNumber(encodedData),
        audio_scales: convertBigIntToNumber(audioScales),
      }),
    });

    if (!response.ok) {
      const errorResponse = await response.json();
      throw new Error(errorResponse.detail);
    }

    const result = await response.json();
    return result.file_path;
  };

  const handleEncode = async () => {
    if (!selectedFile) {
      alert("Please select an audio file first.");
      return;
    }

    setProcessing(true);

    try {
      const convertedAudio = await convertAudio(selectedFile);
      const audioBuffer = new Float32Array(convertedAudio.buffer);
      const session = await prepareInferenceSession();
      const results = await runInference(session, audioBuffer);

      const encodedData = results[session.outputNames[0]].data;
      const audioScales = results[session.outputNames[1]].data;

      const backendFilePath = await sendToBackend(encodedData, audioScales);
      setAudioUrl(`http://127.0.0.1:8000/${backendFilePath}`);
      alert("Success! Audio is ready to play.");
    } catch (error) {
      alertError("An error occurred during processing:", error);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div>
      <h1>EnCodec Task</h1>
      <input type="file" accept="audio/*" onChange={handleFileChange} />
      <button onClick={handleEncode} disabled={processing}>
        {processing ? "Processing..." : "Encode and Send"}
      </button>
      {audioUrl && (
        <div>
          <h2>Decoded Audio</h2>
          <audio src={audioUrl} controls />
        </div>
      )}
    </div>
  );
}

export default App;
