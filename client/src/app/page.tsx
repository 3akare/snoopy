"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { toast } from "sonner";
import Recorder from "@/app/components/Recorder"
import TextDisplay from "@/app/components/TextDisplay"
import ControlDock from "@/app/components/ControlDock"

import * as HandsModule from "@mediapipe/hands";
import * as CameraUtilsModule from "@mediapipe/camera_utils";

const MIN_DETECTION_CONFIDENCE = 0.7;
const MIN_TRACKING_CONFIDENCE = 0.7;
const NUM_HAND_LANDMARKS = 21;
type RecorderState = "default" | "initializing" | "recording" | "ready" | "loading" | "error" | "show-text";

export default function Home() {
    const [state, setState] = useState<RecorderState>("default");
    const [translatedText, setTranslatedText] = useState<string>("");
    const [error, setError] = useState<string>("");
    const [highlightedIndex, setHighlightedIndex] = useState<number>(-1);
    const [isPlaying, setIsPlaying] = useState(false);

    const [handsDetected, setHandsDetected] = useState<number>(0);
    const [keypointsCount, setKeypointsCount] = useState<number>(0);

    const videoRef = useRef<HTMLVideoElement>(null);
    const streamRef = useRef<MediaStream | null>(null);

    const handsRef = useRef<HandsModule.Hands | null>(null);
    const cameraRef = useRef<CameraUtilsModule.Camera | null>(null);
    const recordedKeypointsRef = useRef<number[][]>([]);

    const textContainerRef = useRef<HTMLDivElement>(null);
    const wordsRef = useRef<string[]>([]);

    const onResults = useCallback((results: HandsModule.Results) => {
        if (!videoRef.current) return;
        let handsDetectedThisFrame = 0;
        let userLeftHandKps: number[] = Array(NUM_HAND_LANDMARKS * 3).fill(0.0);
        let userRightHandKps: number[] = Array(NUM_HAND_LANDMARKS * 3).fill(0.0);
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0 && results.multiHandedness) {
            for (let i = 0; i < results.multiHandLandmarks.length; i++) {
                const landmarks = results.multiHandLandmarks[i];
                const handednessEntry = results.multiHandedness[i];
                const handLabel = (handednessEntry as any).label as string;
                const confidence = (handednessEntry as any).score as number;

                if (confidence >= MIN_DETECTION_CONFIDENCE && (handLabel === "Left" || handLabel === "Right")) {
                    handsDetectedThisFrame++;
                    const wristLandmark = landmarks[0];
                    if (!wristLandmark) continue;

                    const wristX = wristLandmark.x;
                    const wristY = wristLandmark.y;
                    const wristZ = wristLandmark.z;
                    const normalizedLandmarks = landmarks.map(lm => [
                        lm.x - wristX,
                        lm.y - wristY,
                        lm.z - wristZ
                    ]).flat();

                    if (handLabel === "Left") {
                        userLeftHandKps = normalizedLandmarks;
                    } else if (handLabel === "Right") {
                        userRightHandKps = normalizedLandmarks;
                    }
                }
            }
        }
        setHandsDetected(handsDetectedThisFrame);
        const currentFrameCombinedKeypoints = [...userLeftHandKps, ...userRightHandKps];
        if (state === "recording") {
            if (currentFrameCombinedKeypoints.some(k => k !== 0.0)) {
                recordedKeypointsRef.current.push(currentFrameCombinedKeypoints);
                setKeypointsCount(recordedKeypointsRef.current.length);
            }
        }
    }, [state]);


    useEffect(() => {
        const initializeHands = async () => {
            try {
                handsRef.current = new HandsModule.Hands({
                    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
                });

                await handsRef.current.setOptions({
                    maxNumHands: 2,
                    modelComplexity: 1,
                    minDetectionConfidence: MIN_DETECTION_CONFIDENCE,
                    minTrackingConfidence: MIN_TRACKING_CONFIDENCE,
                });

                if (handsRef.current) {
                    handsRef.current.onResults(onResults);
                }
                console.log('MediaPipe Hands initialized successfully');
            } catch (error) {
                console.error('Failed to initialize MediaPipe Hands:', error);
                toast.error('Failed to initialize hand detection');
            }
        };
        initializeHands();
        return () => {
            if (handsRef.current && typeof handsRef.current.close === 'function') {
                handsRef.current.close().catch(err => console.error("Error closing MediaPipe Hands:", err));
            }
        };
    }, [onResults]);

    const startRecording = useCallback(async () => {
        setError(""); setTranslatedText(""); setHighlightedIndex(-1); setIsPlaying(false);
        setHandsDetected(0); setKeypointsCount(0);
        recordedKeypointsRef.current = [];

        setState("initializing");

        try {
            console.log('Requesting camera access...');
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640, min: 480 },
                    height: { ideal: 480, min: 360 },
                    frameRate: { ideal: 30, min: 15 }
                },
                audio: false
            });
            streamRef.current = stream;

            let attempts = 0;
            const maxAttempts = 50;
            while (!videoRef.current && attempts < maxAttempts) {
                await new Promise(resolve => setTimeout(resolve, 100));
                attempts++;
            }

            if (!videoRef.current) {
                throw new Error("Video element failed to initialize in time. Please ensure camera access and try again.");
            }

            videoRef.current.srcObject = stream;

            await new Promise<void>((resolve, reject) => {
                const timeoutId = setTimeout(() => reject(new Error('Video metadata load timeout')), 10000);
                if (!videoRef.current) { clearTimeout(timeoutId); reject(new Error("Video element became null.")); return; }
                if (videoRef.current.readyState >= HTMLMediaElement.HAVE_METADATA) { clearTimeout(timeoutId); resolve(); return; }
                videoRef.current.onloadedmetadata = () => { clearTimeout(timeoutId); resolve(); };
                videoRef.current.onerror = () => { clearTimeout(timeoutId); reject(new Error('Video element error loading metadata.')); };
            });

            if (!videoRef.current) throw new Error("Video element became null before play.");
            await videoRef.current.play();
            console.log('Video stream started:', { width: videoRef.current?.videoWidth, height: videoRef.current?.videoHeight });
            await new Promise(resolve => setTimeout(resolve, 200));

            if (!handsRef.current) throw new Error('MediaPipe Hands not ready. Initialization failed.');
            if (!videoRef.current) throw new Error('Video element unavailable for MediaPipe camera.');

            cameraRef.current = new CameraUtilsModule.Camera(videoRef.current, {
                onFrame: async () => {
                    if (handsRef.current && videoRef.current && videoRef.current.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
                        try { await handsRef.current.send({ image: videoRef.current }); }
                        catch (frameError) { console.error('Error sending frame to MediaPipe:', frameError); }
                    }
                },
            });

            await cameraRef.current.start();
            await new Promise(resolve => setTimeout(resolve, 100));

            setState("recording");
            toast.info("Recording started!", { duration: 3000 });

        } catch (err) {
            const errorMessage = (err instanceof Error ? err.message : String(err)) || "Unknown setup error.";
            console.error('Start recording error:', err); setError(errorMessage); setState("error");
            toast.error(`Error: ${errorMessage}`, { duration: 5000 });

            if (streamRef.current) { streamRef.current.getTracks().forEach(track => track.stop()); streamRef.current = null; }
            if (videoRef.current) videoRef.current.srcObject = null;
            if (cameraRef.current) { cameraRef.current.stop(); cameraRef.current = null; }
        }
    }, []);


    const stopRecording = useCallback(() => {
        console.log('Stopping recording and turning off camera...');

        if (cameraRef.current) {
            cameraRef.current.stop();
            cameraRef.current = null;
        }

        if (streamRef.current) {
            streamRef.current.getTracks().forEach(track => track.stop());
            streamRef.current = null;
        }
        
        if (videoRef.current) {
            videoRef.current.srcObject = null;
            videoRef.current.load();
        }

        setState("ready");
        console.log(`Recording stopped. Total keypoint frames: ${recordedKeypointsRef.current.length}`);
    }, []);

    const sendRecording = useCallback(async () => {
        if (recordedKeypointsRef.current.length === 0) {
            toast.error("No recording/keypoints to send.", { duration: 3000 }); resetRecorder(); return;
        }
        const hasValidKeypoints = recordedKeypointsRef.current.some(f => f.some(v => Math.abs(v) > 0.001));
        if (!hasValidKeypoints) {
            toast.error("No valid hand gestures detected.", { duration: 3000 }); resetRecorder(); return;
        }
        setState("loading");
        try {
            console.log("====>", recordedKeypointsRef.current)
            const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080/predict_gesture";
            const res = await fetch(apiUrl, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ keypoints: recordedKeypointsRef.current }),
            });
            if (!res.ok) {
                const errorData = await res.json().catch(() => ({ message: "Unknown server error" }));
                throw new Error(errorData.message || "Server responded with an error.");
            }
            const data = await res.json();
            const newText = data.translatedText || "No translation.";
            setTranslatedText(newText); wordsRef.current = newText.split(/\s+/).filter(Boolean);
            setState("show-text"); toast.success("Gesture recognized!", { duration: 2000 });
        } catch (err: any) {
            console.error("Error sending keypoints for prediction:", err);
            setError(err.message || "Failed to process keypoints. Please try again.");
            setState("error");
            toast.error("Prediction failed. " + (err.message || "Unknown error."), { duration: 5000 });
            resetRecorder();
        }
    }, []);

    const playTranslatedText = useCallback(() => {
        if (!translatedText || typeof window.speechSynthesis === 'undefined') return;
        if (isPlaying) { window.speechSynthesis.cancel(); setIsPlaying(false); setHighlightedIndex(-1); return; }
        setIsPlaying(true); setHighlightedIndex(0);
        const utterance = new SpeechSynthesisUtterance(translatedText);
        utterance.rate = 1.0; utterance.pitch = 1.0;
        utterance.onboundary = (event) => {
            if (event.name === "word" && wordsRef.current && event.charIndex !== undefined) {
                let currentWordIndex = 0;
                let charCount = 0;
                for (let i = 0; i < wordsRef.current.length; i++) {
                    charCount += wordsRef.current[i].length + 1;
                    if (event.charIndex < charCount) {
                        currentWordIndex = i;
                        break;
                    }
                }
                setHighlightedIndex(currentWordIndex);
            }
        };
        utterance.onend = () => { setIsPlaying(false); setHighlightedIndex(-1); };
        utterance.onerror = (event) => {
            console.error("SpeechSynthesisUtterance error:", event.error);
            setIsPlaying(false); setHighlightedIndex(-1);
            toast.error("Text-to-speech failed.", { duration: 3000 });
        };

        let attempts = 0;
        const speakWhenVoicesReady = () => {
            attempts++; const voices = window.speechSynthesis.getVoices();
            if (voices.length > 0) {
                const bestVoice = voices.find(voice => voice.lang === 'en-US');
                if (bestVoice) utterance.voice = bestVoice;
                window.speechSynthesis.speak(utterance);
            }
            else if (attempts < 10) { setTimeout(speakWhenVoicesReady, 100); }
            else { console.warn("No voices loaded after multiple attempts. Speaking with default voice."); window.speechSynthesis.speak(utterance); }
        };
        if (window.speechSynthesis.getVoices().length === 0) {
            window.speechSynthesis.onvoiceschanged = () => speakWhenVoicesReady();
            speakWhenVoicesReady();
        } else {
            speakWhenVoicesReady();
        }
    }, [isPlaying, translatedText]);

    const copyToClipboard = useCallback(() => {
        if (!translatedText) return;
        navigator.clipboard.writeText(translatedText)
            .then(() => toast.success("Copied to clipboard", { description: "Text has been copied to your clipboard", duration: 2000, className: "!text-blue-600" }))
            .catch((err) => { console.error("Failed to copy text:", err); toast.error("Failed to copy", { description: "Could not copy text to clipboard", duration: 2000, className: "!text-blue-600" }); });
    }, [translatedText]);

    const resetRecorder = useCallback(() => {
        if (typeof window.speechSynthesis !== 'undefined') window.speechSynthesis.cancel();
        if (cameraRef.current) { cameraRef.current.stop(); cameraRef.current = null; }
        if (streamRef.current) { streamRef.current.getTracks().forEach(track => track.stop()); streamRef.current = null; }
        if (videoRef.current) { videoRef.current.srcObject = null; videoRef.current.load(); }

        recordedKeypointsRef.current = [];
        setTranslatedText("");
        setError("");
        setHighlightedIndex(-1);
        setIsPlaying(false);
        setHandsDetected(0);
        setKeypointsCount(0);
        setState("default");
    }, []);

    return (
        <div className="relative flex flex-col w-full h-[calc(100vh-2rem)] max-h-screen overflow-hidden">
            <div className="flex-1 p-4 overflow-auto">
                {state !== "show-text" ? (
                    // Changed onRetry to onReset for Recorder component as per common flow
                    <Recorder state={state} onReset={resetRecorder} onRetry={resetRecorder} />
                ) : (
                    <TextDisplay ref={textContainerRef} translatedText={translatedText} highlightedIndex={highlightedIndex} words={wordsRef.current} onCopyToClipboard={copyToClipboard} />
                )}
                {state === "error" && error && (
                    <div className="flex flex-col items-center justify-center mt-4">
                        <p className="text-red-600 mb-4 text-center">{error}</p>
                        <button onClick={resetRecorder} className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded-md text-sm">Reset</button>
                    </div>
                )}
                {/* Debug / Status Overlay */}
                {(state === "recording" || state === "initializing") && ( // Show only when recording or initializing
                    <div className="fixed top-4 left-4 bg-black bg-opacity-80 text-white p-3 rounded text-sm z-50 font-mono">
                        <div>Status: <span className="text-green-400">{state}</span></div>
                        <div>Hands Detected: <span className="text-blue-400">{handsDetected}</span></div>
                        <div>Frames: <span className="text-yellow-400">{keypointsCount}</span></div>
                        <div>Stream: <span className={streamRef.current ? "text-green-400" : "text-red-400"}>{streamRef.current ? "Active" : "Off"}</span></div>
                        <div>MediaPipe: <span className={handsRef.current ? "text-green-400" : "text-red-400"}>{handsRef.current ? "Ready" : "Init..."}</span></div>
                    </div>
                )}
            </div>
            {/* Small preview video - visible only during active camera states */}
            {(state === 'initializing' || state === 'recording') && ( // Condition refined for visibility
                <div className="absolute bottom-20 right-4 z-10 md:block">
                    <div className="md:w-48 md:h-32 xl:w-72 xl:h-48 overflow-hidden rounded-lg border border-blue-200 shadow-lg bg-black">
                        <video
                            ref={videoRef}
                            className="w-full h-full object-cover transform -scale-x-100"
                            muted
                            autoPlay
                            playsInline
                        />
                    </div>
                </div>
            )}
            <ControlDock state={state} isTextAvailable={!!translatedText} isPlaying={isPlaying} onStartRecording={startRecording} onStopRecording={stopRecording} onResetRecorder={resetRecorder} onSendRecording={sendRecording} onPlayTranslatedText={playTranslatedText} />
        </div>
    )
}