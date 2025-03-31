import {useCallback, useRef, useState} from "react";
import {toast} from "sonner";
import Recorder from "./components/Recorder";
import TextDisplay from "./components/TextDisplay";
import ControlDock from "./components/ControlDock";

type RecorderState = "default" | "recording" | "ready" | "loading" | "error" | "show-text";

export default function App() {
    const [state, setState] = useState<RecorderState>("default");
    const [translatedText, setTranslatedText] = useState<string>("");
    const [error, setError] = useState<string>("");
    const [highlightedIndex, setHighlightedIndex] = useState<number>(-1);
    const [isPlaying, setIsPlaying] = useState(false);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const recordedBlobRef = useRef<Blob | null>(null);
    const textContainerRef = useRef<HTMLDivElement>(null);
    const wordsRef = useRef<string[]>([]);

    // Start recording
    const startRecording = useCallback(async () => {
        try {
            chunksRef.current = [];
            const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false});
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) chunksRef.current.push(e.data);
            };

            mediaRecorder.onstop = () => {
                recordedBlobRef.current = new Blob(chunksRef.current, {type: "video/webm"});
                // Stop all tracks from the stream
                stream.getTracks().forEach((track) => track.stop());
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
            // Simulate API call with a timeout
            // In a real app, you would upload the video to your server here
            await new Promise((resolve) => setTimeout(resolve, 500));

            // Simulate receiving translated text
            const mockTranslatedText = `Hello, my name is David and this is my implementation of a Nigerian Sign Language to speech system.`;
            setTranslatedText(mockTranslatedText);
            wordsRef.current = mockTranslatedText.split(/\s+/);
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
    );
}