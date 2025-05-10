"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { toast } from "sonner";
import Recorder from "@/app/components/Recorder"
import TextDisplay from "@/app/components/TextDisplay"
import ControlDock from "@/app/components/ControlDock"

type RecorderState = "default" | "recording" | "ready" | "loading" | "error" | "show-text"

export default function Home() {
    const [state, setState] = useState<RecorderState>("default");
    const [translatedText, setTranslatedText] = useState<string>("");
    const [error, setError] = useState<string>("");
    const [highlightedIndex, setHighlightedIndex] = useState<number>(-1);
    const [isPlaying, setIsPlaying] = useState(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null)
    const chunksRef = useRef<Blob[]>([]);
    const recordedBlobRef = useRef<Blob | null>(null);
    const textContainerRef = useRef<HTMLDivElement>(null);
    const wordsRef = useRef<string[]>([]);
    const streamRef = useRef<MediaStream | null>(null);

    useEffect(() => {
        if (state === "recording" && videoRef.current && streamRef.current) {
            videoRef.current.srcObject = streamRef.current;
            videoRef.current.play().catch(err => console.error("Video play failed:", err));
        }
    }, [state]);


    // Start recording
    const startRecording = useCallback(async () => {
        try {
            chunksRef.current = [];
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            streamRef.current = stream;

            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            mediaRecorder.onstop = () => {
                recordedBlobRef.current = new Blob(chunksRef.current, { type: "video/webm" });
                stream.getTracks().forEach((track) => track.stop());

                // Clear video
                if (videoRef.current) videoRef.current.srcObject = null;

                setState("ready");
            };

            mediaRecorder.start();
            setState("recording");
        } catch (err) {
            console.error("Error starting recording:", err);
            setError("Could not access camera. Please check permissions and try again.");
            setState("error");
        }
    }, []);



    // Stop recording
    const stopRecording = useCallback(() => {
        if (mediaRecorderRef.current && state === "recording") mediaRecorderRef.current.stop();
    }, [state]);

    // Send recording to server
    const sendRecording = useCallback(async () => {
        if (!recordedBlobRef.current) return;

        setState("loading");
        try {
            const formData = new FormData();
            formData.append("video", recordedBlobRef.current, "recording.webm");
            const res = await fetch("http://localhost:8080/upload", {
                method: "POST",
                body: formData,
            });

            if (!res.ok) throw new Error("Upload failed");
            const data = await res.json();
            setTranslatedText(data.translatedText);
            const mockTranslatedText = `Hello, my name is David and this is my implementation of a Nigerian Sign Language to speech system.`;
            wordsRef.current = data.translatedText.split(/\s+/);
            setState("show-text");
        } catch (err) {
            console.error("Error sending recording:", err);
            setError("Failed to upload video. Please try again.");
            setState("error");
        }
    }, []);

    const playTranslatedText = useCallback(() => {
        if (!translatedText) return;

        if (isPlaying) {
            window.speechSynthesis.cancel();
            setIsPlaying(false);
            setHighlightedIndex(-1);
            return;
        }

        const words = translatedText.split(" ");
        setIsPlaying(true);
        setHighlightedIndex(0);

        const utterance = new SpeechSynthesisUtterance(translatedText);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;

        // Select the best available voice
        const voices = window.speechSynthesis.getVoices();
        const bestVoice = voices.find(voice => voice.lang === 'en-US'); // You can adjust this to find a more fluent voice

        if (bestVoice) utterance.voice = bestVoice;
        let wordIndex = 0;

        // Sync highlighting with spoken words
        utterance.onboundary = (event) => {
            if (event.name === "word" && wordIndex < words.length) {
                setHighlightedIndex(wordIndex);
                wordIndex++;
            }
        };

        utterance.onend = () => {
            setIsPlaying(false);
            setHighlightedIndex(-1);
        };

        utterance.onerror = () => {
            setIsPlaying(false);
            setHighlightedIndex(-1);
        };

        // Ensure the voices are loaded (in case it's not immediately available)
        if (window.speechSynthesis.getVoices().length === 0) {
            window.speechSynthesis.onvoiceschanged = () => {
                window.speechSynthesis.speak(utterance);
            };
        } else {
            window.speechSynthesis.speak(utterance);
        }
    }, [isPlaying, translatedText]);

    // Copy translated text to clipboard
    const copyToClipboard = useCallback(() => {
        if (!translatedText) return;

        navigator.clipboard
            .writeText(translatedText)
            .then(() => {
                toast.success("Copied to clipboard", {
                    description: "Text has been copied to your clipboard",
                    duration: 2000,
                    className: "bg-green-900 text-white",
                });
            })
            .catch((err) => {
                console.error("Failed to copy text:", err);
                toast.error("Failed to copy", {
                    description: "Could not copy text to clipboard",
                    duration: 2000,
                    className: "bg-red-500 text-white",
                });
            });
    }, [translatedText]);

    // Reset to default state
    const resetRecorder = useCallback(() => {
        // Stop any ongoing speech synthesis
        window.speechSynthesis.cancel();

        recordedBlobRef.current = null;
        setTranslatedText("");
        setError("");
        setHighlightedIndex(-1);
        setIsPlaying(false);
        setState("default");
    }, []);

    return (
        <div className="relative flex flex-col w-full h-[calc(100vh-2rem)] max-h-screen overflow-hidden">
            {/* Main Display Area */}
            <div className="flex-1 p-4 overflow-auto">
                {state !== "show-text" ? (
                    <Recorder
                        state={state}
                        onReset={() => setState("default")}
                        onRetry={() => setState("ready")}
                    />
                ) : (
                    <TextDisplay
                        ref={textContainerRef}
                        translatedText={translatedText}
                        highlightedIndex={highlightedIndex}
                        words={wordsRef.current}
                        onCopyToClipboard={copyToClipboard}
                    />
                )}
                {state === "error" && (
                    <div className="flex flex-col items-center justify-center mt-4">
                        <p className="text-gray-600 mb-4">{error}</p>
                        <button onClick={() => setState("ready")} className="px-4 py-2 bg-gray-200 rounded-md text-sm">
                            Retry
                        </button>
                    </div>
                )}
            </div>
            {state === "recording" && (
                <div className="absolute bottom-20 right-4 z-10 hidden md:block">
                    <div className="md:w-48 md:h-32 xl:w-72 xl:h-48 overflow-hidden rounded-lg border border-green-900/50 shadow-lg bg-black">
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

            {/* Dock (Control) Component - Fixed at bottom */}
            <ControlDock
                state={state}
                isTextAvailable={!!translatedText}
                isPlaying={isPlaying}
                onStartRecording={startRecording}
                onStopRecording={stopRecording}
                onResetRecorder={resetRecorder}
                onSendRecording={sendRecording}
                onPlayTranslatedText={playTranslatedText}
            />
        </div>
    )
}
