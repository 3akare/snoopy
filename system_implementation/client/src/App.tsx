import {useRef, useState} from "react"
import {Copy, Loader2, Play, Send, Square, Trash, Video, X} from "lucide-react"
import {cn} from "@/lib/utils.ts"
import {toast} from "sonner"

type RecorderState = "default" | "recording" | "ready" | "loading" | "error" | "show-text"

export default function App() {
    const [state, setState] = useState<RecorderState>("default")
    const [translatedText, setTranslatedText] = useState<string>("")
    const [error, setError] = useState<string>("")
    const [highlightedIndex, setHighlightedIndex] = useState<number>(-1)
    const [isPlaying, setIsPlaying] = useState(false)
    const mediaRecorderRef = useRef<MediaRecorder | null>(null)
    const chunksRef = useRef<Blob[]>([])
    const recordedBlobRef = useRef<Blob | null>(null)
    const textContainerRef = useRef<HTMLDivElement>(null)
    const wordsRef = useRef<string[]>([])

    // Start recording
    const startRecording = async () => {
        try {
            chunksRef.current = []
            const stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false})
            const mediaRecorder = new MediaRecorder(stream)
            mediaRecorderRef.current = mediaRecorder
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data)
                }
            }

            mediaRecorder.onstop = () => {
                const blob = new Blob(chunksRef.current, {type: "video/webm"})
                recordedBlobRef.current = blob
                // Stop all tracks from the stream
                stream.getTracks().forEach((track) => track.stop())
            }

            mediaRecorder.start()
            setState("recording")
        } catch (err) {
            console.error("Error starting recording:", err)
            setError("Could not access camera. Please check permissions and try again.")
            setState("error")
        }
    }

    // Stop recording
    const stopRecording = () => {
        if (mediaRecorderRef.current && state === "recording") {
            mediaRecorderRef.current.stop()
            setState("ready")
        }
    }

    // Display the recorded video
    // const getRecordedVideoUrl = () => {
    //     if (recordedBlobRef.current) {
    //         return URL.createObjectURL(recordedBlobRef.current);
    //     }
    //     return '';
    // };

    // Send recording to server
    const sendRecording = async () => {
        if (!recordedBlobRef.current) return
        setState("loading")
        try {
            // Simulate API call with a timeout
            // In a real app, you would upload the video to your server here
            await new Promise((resolve) => setTimeout(resolve, 2000))

            // Simulate receiving translated text
            const mockTranslatedText = `MediaPipe is a powerful and readily available tool developed by Google. It can track the movement of hands and other body parts in real-time from video. This is very useful for sign language recognition because it gives us numerical data about the position and movement of the hands, which our computer model can learn from. It's like giving the computer a set of coordinates for the hands as they form a sign.`
            setTranslatedText(mockTranslatedText)
            wordsRef.current = mockTranslatedText.split(/\s+/)
            setState("show-text")
        } catch (err) {
            console.error("Error sending recording:", err)
            setError("Failed to upload video. Please try again.")
            setState("error")
        }
    }

    const playTranslatedText = () => {
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
        if (bestVoice) {
            utterance.voice = bestVoice;
        }

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
    };


    // Copy translated text to clipboard
    const copyToClipboard = () => {
        if (!translatedText) return

        navigator.clipboard
            .writeText(translatedText)
            .then(() => {
                toast.success("Copied to clipboard", {
                    description: "Text has been copied to your clipboard",
                    duration: 2000,
                    className: "bg-[rgb(0,31,15)] text-white",
                })
            })
            .catch((err) => {
                console.error("Failed to copy text:", err)
                toast.error("Failed to copy", {
                    description: "Could not copy text to clipboard",
                    duration: 2000,
                    className: "bg-red-500 text-white",
                })
            })
    }

    // Reset to default state
    const resetRecorder = () => {
        // Stop any ongoing speech synthesis
        window.speechSynthesis.cancel()

        recordedBlobRef.current = null
        setTranslatedText("")
        setError("")
        setHighlightedIndex(-1)
        setIsPlaying(false)
        setState("default")
    }

    // Render display content based on state
    const renderDisplay = () => {
        switch (state) {
            case "default":
                return (
                    <div className="flex flex-col items-center justify-center h-full">
                        <p className="text-xl text-center">Click Video to get started.</p>
                    </div>
                )

            case "recording":
                return (
                    <div className="flex flex-col items-center justify-center h-full">
                        <div className="text-center">
                            <div
                                className="inline-flex items-center gap-2 bg-red-100 text-red-600 px-4 py-2 rounded-full">
                                <div className="w-3 h-3 bg-red-600 rounded-full animate-pulse"></div>
                                <p className="font-medium">Recording...</p>
                            </div>
                        </div>
                    </div>
                )

            case "ready":
                return (
                    <div className="flex flex-col items-center justify-center h-full">
                        <p className="text-xl text-center mb-4">Click Send to upload the video or Delete to discard the
                            recording.</p>
                    </div>
                )

            case "loading":
                return (
                    <div className="flex flex-col items-center justify-center h-full">
                        <div className="text-center">
                            <Loader2 className="w-10 h-10 text-[rgb(0,31,15)] animate-spin mx-auto mb-4"/>
                        </div>
                    </div>
                )

            case "error":
                return (
                    <div className="flex flex-col items-center justify-center h-full">
                        <div className="text-center">
                            <p className="text-red-500 text-xl mb-2">An error occurred</p>
                            <p className="text-gray-600 mb-4">{error}</p>
                            <div className="flex gap-2 justify-center">
                                <button onClick={() => setState("ready")}
                                        className="px-4 py-2 bg-gray-200 rounded-md text-sm">
                                    Retry
                                </button>
                                <button onClick={resetRecorder}
                                        className="px-4 py-2 bg-red-100 text-red-600 rounded-md text-sm">
                                    Delete
                                </button>
                            </div>
                        </div>
                    </div>
                )

            case "show-text":
                return (
                    <div className="flex flex-col h-full p-4">
                        <div ref={textContainerRef} className="relative flex-1 p-6 overflow-auto">
                            <div className="text-xl leading-relaxed">
                                {wordsRef.current.map((word, index) => (
                                    <span
                                        key={index}
                                        className={cn(
                                            "transition-colors duration-200",
                                            index === highlightedIndex ? "text-gray-900 font-medium" : "text-gray-400",
                                        )}
                                    >
                    {word}{" "}
                  </span>
                                ))}
                            </div>
                            <button
                                onClick={copyToClipboard}
                                className="absolute top-3 right-3 p-2 rounded-md hover:bg-gray-100"
                                aria-label="Copy to clipboard"
                            >
                                <Copy className="w-5 h-5"/>
                            </button>
                        </div>
                        {/*{recordedBlobRef.current && (*/}
                        {/*    <div>*/}
                        {/*        <h3>Recorded Video:</h3>*/}
                        {/*        <video width="640" height="480" controls>*/}
                        {/*            <source src={getRecordedVideoUrl()} type="video/webm" />*/}
                        {/*            Your browser does not support the video tag.*/}
                        {/*        </video>*/}
                        {/*    </div>*/}
                        {/*)}*/}
                    </div>
                )

            default:
                return null
        }
    }

    // Determine which icon to show for the first button based on state
    const renderFirstButtonIcon = () => {
        if (state === "recording") {
            return <Square className="w-6 h-6"/>
        } else if (state === "ready") {
            return <Trash className="w-6 h-6"/>
        } else {
            return <Video className="w-6 h-6"/>
        }
    }

    // First button action based on state
    const handleFirstButtonClick = () => {
        if (state === "recording") {
            stopRecording()
        } else if (state === "ready") {
            resetRecorder()
        } else {
            startRecording()
        }
    }

    return (
        <div className="relative flex flex-col w-full h-[calc(100vh-2rem)] max-h-screen overflow-hidden">
            {/* Main Display Area */}
            <div className="flex-1 p-4 overflow-auto">{renderDisplay()}</div>

            {/* Dock (Control) Component - Fixed at bottom */}
            <div className="sticky bottom-0 left-0 right-0 flex justify-center pb-6 pt-2">
                <div
                    className="bg-white/80 backdrop-blur-md border rounded-full px-6 py-3 shadow-lg flex items-center gap-8">
                    {/* First Button (Camera/Stop/Trash) */}
                    <button
                        onClick={handleFirstButtonClick}
                        disabled={["loading", "show-text"].includes(state)}
                        className={cn(
                            "p-3 rounded-full transition-all transform hover:scale-110",
                            state === "recording"
                                ? "bg-red-500 text-white shadow-md"
                                : state === "ready"
                                    ? "bg-gray-200 text-gray-700 shadow-md"
                                    : "bg-gray-100 hover:bg-gray-200",
                            ["loading", "show-text"].includes(state) && "opacity-50 cursor-not-allowed",
                        )}
                        aria-label={
                            state === "recording" ? "Stop recording" : state === "ready" ? "Delete recording" : "Start recording"
                        }
                    >
                        {renderFirstButtonIcon()}
                    </button>

                    {/* Play Button */}
                    <button
                        onClick={playTranslatedText}
                        disabled={!translatedText || ["recording", "loading"].includes(state)}
                        className={cn(
                            "p-3 rounded-full transition-all transform hover:scale-110",
                            isPlaying ? "bg-[rgb(0,31,15)] text-white shadow-md" : "bg-gray-100 hover:bg-gray-200",
                            (!translatedText || ["recording", "loading"].includes(state)) && "opacity-50 cursor-not-allowed",
                        )}
                        aria-label={isPlaying ? "Restart speech" : "Play translated text"}
                    >
                        <Play className="w-6 h-6"/>
                    </button>

                    {/* Send Button */}
                    <button
                        onClick={state === "ready" ? sendRecording : resetRecorder}
                        disabled={["recording", "loading", "default"].includes(state)}
                        className={cn(
                            "p-3 rounded-full transition-all transform hover:scale-110",
                            state === "ready"
                                ? "bg-[rgb(0,31,15)] text-white shadow-md"
                                : state === "show-text" || state === "error"
                                    ? "bg-gray-100 hover:bg-gray-200"
                                    : "bg-gray-100",
                            ["recording", "loading", "default"].includes(state) && "opacity-50 cursor-not-allowed",
                        )}
                        aria-label={state === "ready" ? "Send recording" : "Reset"}
                    >
                        {state === "ready" ? <Send className="w-6 h-6"/> : <X className="w-6 h-6"/>}
                    </button>
                </div>
            </div>
        </div>
    )
}

