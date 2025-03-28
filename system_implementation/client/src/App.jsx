import {AnimatePresence, motion} from "framer-motion"
import {useEffect, useRef, useState} from "react";
import { Button } from "@/components/ui/button"
import { Send, Video, Square, Trash2 } from "lucide-react"
import {cn} from "@/lib/utils.js";
import Profile from "@/components/Profile.jsx";

export default function App() {
    const [isRecording, setIsRecording] = useState(false)
    const [recordedVideo, setRecordedVideo] = useState(null)
    const [videoThumbnail, setVideoThumbnail] = useState(null)
    const [isProcessing, setIsProcessing] = useState(false)
    const [messages, setMessages] = useState([
        {
            id: "welcome-0",
            content: "Record a video message to get started.",
            sender: "assistant",
            timestamp: new Date(),
        }
    ])

    const messagesEndRef = useRef(null)
    const mediaRecorderRef = useRef(null)
    const videoChunksRef = useRef([])
    const streamRef = useRef(null)

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({behavior: "smooth"})
    }, [messages])

    // Clean up media stream on unmount
    useEffect(() => {
        return () => {
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track) => track.stop())
            }
        }
    }, [])

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: true,
            })

            streamRef.current = stream
            const mediaRecorder = new MediaRecorder(stream)
            mediaRecorderRef.current = mediaRecorder
            videoChunksRef.current = []

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    videoChunksRef.current.push(e.data)
                }
            }

            mediaRecorder.onstop = () => {
                const videoBlob = new Blob(videoChunksRef.current, { type: "video/webm" })
                setRecordedVideo(videoBlob)
                generateThumbnail(videoBlob)
            }

            mediaRecorder.start()
            setIsRecording(true)
        } catch (error) {
            console.error("Error accessing media devices:", error)
        }
    }

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop()
            setIsRecording(false)

            // Stop all tracks
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track) => track.stop())
                streamRef.current = null
            }
        }
    }

    const generateThumbnail = (videoBlob) => {
        const video = document.createElement("video")
        video.autoplay = false
        video.muted = true
        video.src = URL.createObjectURL(videoBlob)

        video.onloadeddata = () => {
            video.currentTime = 0.5 // Seek to 0.5 seconds
        }

        video.onseeked = () => {
            const canvas = document.createElement("canvas")
            canvas.width = video.videoWidth
            canvas.height = video.videoHeight

            const ctx = canvas.getContext("2d")
            if (ctx) {
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
                const thumbnailUrl = canvas.toDataURL("image/jpeg")
                setVideoThumbnail(thumbnailUrl)
            }

            URL.revokeObjectURL(video.src)
        }
    }

    const clearRecording = () => {
        setRecordedVideo(null)
        setVideoThumbnail(null)
    }

    const sendVideoMessage = async () => {
        if (!recordedVideo) return

        setIsProcessing(true)

        // Add user message with video thumbnail
        const userMessage = {
            id: Date.now().toString(),
            content: "Video message",
            sender: "user",
            timestamp: new Date(),
            videoThumbnail: videoThumbnail || undefined,
        }

        setMessages((prev) => [...prev, userMessage])

        // In a real app, you would upload the video to your server here
        // For demo purposes, we'll simulate a server response

        // Simulate server processing time (1-3 seconds)
        const processingTime = 1000 + Math.random() * 2000

        setTimeout(() => {
            // Add assistant response
            const assistantMessage = {
                id: (Date.now() + 1).toString(),
                content: getRandomResponse(),
                sender: "assistant",
                timestamp: new Date(),
            }

            setMessages((prev) => [...prev, assistantMessage])
            setIsProcessing(false)
            clearRecording()
        }, processingTime)
    }

    const getRandomResponse = () => {
        const responses = [
            "I've analyzed your video. It looks like you're asking about our services. We offer a range of AI-powered solutions for businesses of all sizes.",
            "Thanks for your video message. I understand you're interested in our pricing plans. Our basic plan starts at $9.99/month and includes all essential features.",
            "I see you're having trouble with the application. Have you tried clearing your browser cache and restarting the app? That often resolves this issue.",
            "Thank you for sharing your feedback! We're constantly working to improve our platform based on user suggestions like yours.",
            "I've processed your video request. The information you're looking for can be found in our documentation at docs.example.com.",
        ]

        return responses[Math.floor(Math.random() * responses.length)]
    }

    return (
        <div className={"flex flex-col h-screen bg-black text-white"}>
            {/* header */}
            <header
                className="border-b border-zinc-800 p-4 sticky top-0 z-10 bg-black">
                <nav className={"container mx-auto max-w-5xl flex items-center justify-between"}>
                    <h1 className={"text-lg font-semibold"}>Snoopy</h1>
                    <div className="flex items-center">
                        <Profile src={"https://github.com/shadcn.png"} alt={"@you"} fallback={"DB"}></Profile>
                    </div>
                </nav>
            </header>
            {/* chat container */}
            <div className={"flex-grow overflow-y-auto p-4 sm:p-6 space-y-6"}>
                <AnimatePresence initial={false}>
                    {messages.map((message) => (
                        <motion.div
                            key={message.id}
                            initial={{opacity: 0, y: 20}}
                            animate={{opacity: 1, y: 0}}
                            transition={{duration: 0.3}}
                            className={cn(
                                "flex items-start gap-3 max-w-5xl mx-auto",
                                message.sender === "user" ? "justify-end" : "justify-start",
                            )}>
                            {message.sender === "assistant" && (
                                <Profile src={"https://github.com/shadcn.png"} alt={"@you"} fallback={"DB"}></Profile>
                            )}
                            <div
                                className={cn(
                                    "rounded-2xl px-4 py-3 max-w-[85%]",
                                    message.sender === "user" ? "bg-emerald-600" : "bg-zinc-700",
                                )}
                            >
                                {message.videoThumbnail ? (
                                    <div className="space-y-2">
                                        <div className="relative rounded-lg overflow-hidden aspect-video w-64 sm:w-80">
                                            <img
                                                src={message.videoThumbnail || "/placeholder.svg"}
                                                alt="Video thumbnail"
                                                className="w-full h-full object-cover"
                                            />
                                            <div
                                                className="absolute inset-0 bg-black bg-opacity-20 flex items-center justify-center">
                                                <div
                                                    className="w-12 h-12 rounded-full bg-white bg-opacity-30 flex items-center justify-center">
                                                    <div className="w-3 h-3 bg-white rounded-sm"></div>
                                                </div>
                                            </div>
                                        </div>
                                        <p className="text-xs opacity-70">Video message</p>
                                    </div>
                                ) : (
                                    <p>{message.content}</p>
                                )}

                                <p className="text-xs opacity-50 mt-1">
                                    {message.timestamp.toLocaleTimeString([], {
                                        hour: "2-digit",
                                        minute: "2-digit",
                                    })}
                                </p>
                            </div>

                            {message.sender === "user" && (
                                <Profile src={null} alt={"@you"} fallback={"DB"}></Profile>
                            )}
                        </motion.div>
                    ))}
                </AnimatePresence>
                <div ref={messagesEndRef}/>
            </div>
            <div className="border-t border-zinc-800 p-4 sm:p-6">
                <div className="max-w-5xl mx-auto">
                    <div className="flex items-center gap-3">
                        {/* Recording controls */}
                        {!recordedVideo ? (
                            <Button
                                variant={isRecording ? "destructive" : "outline"}
                                size="icon"
                                onClick={isRecording ? stopRecording : startRecording}
                                className={cn("rounded-full h-12 w-12 flex-shrink-0 bg-zinc-900 border-zinc-900", isRecording && "relative")}
                                disabled={isProcessing}
                            >
                                {isRecording ? <Square className="h-5 w-5"/> : <Video className="h-5 w-5"/>}

                                {isRecording && (
                                    <span
                                        className="absolute inset-0 rounded-full animate-ping bg-red-500 opacity-75"></span>
                                )}
                            </Button>
                        ) : (
                            <Button
                                variant="outline"
                                size="icon"
                                onClick={clearRecording}
                                className="rounded-full h-12 w-12 flex-shrink-0 bg-zinc-900 border-zinc-900"
                                disabled={isProcessing}
                            >
                                <Trash2 className="h-5 w-5"/>
                            </Button>
                        )}

                        {/* Video thumbnail or placeholder */}
                        <div className="flex-grow bg-zinc-800 rounded-2xl p-3 min-h-[60px] flex items-center">
                            {videoThumbnail ? (
                                <div className="flex items-center gap-3">
                                    <div className="relative rounded-md overflow-hidden h-10 w-16">
                                        <img
                                            src={videoThumbnail || "/placeholder.svg"}
                                            alt="Video thumbnail"
                                            className="w-full h-full object-cover"
                                        />
                                    </div>
                                    <span className="text-zinc-400 text-sm">Video ready to send</span>
                                </div>
                            ) : (
                                <span className="text-zinc-400 text-sm">
                  {isRecording ? "Recording..." : "Click the microphone button to start recording"}
                </span>
                            )}
                        </div>

                        {/* Send button */}
                        <Button
                            type="button"
                            size="icon"
                            disabled={!recordedVideo || isProcessing}
                            onClick={sendVideoMessage}
                            className={cn("rounded-full h-12 w-12 flex-shrink-0", !recordedVideo && "opacity-50 cursor-not-allowed")}
                        >
                            <Send className="h-5 w-5"/>
                        </Button>
                    </div>

                    {/* Processing indicator */}
                    {isProcessing &&
                        <div className="mt-2 text-center text-sm text-zinc-400">Processing your video...</div>}
                </div>
            </div>
        </div>
    );
}