import {ControlDockProps} from "@/types/ControlDocksProps";
import {Play, Send, Square, Trash, Video, X} from "lucide-react";
import {cn} from "@/lib/utils";

const ControlDock = ({
                         state,
                         isTextAvailable,
                         isPlaying,
                         onStartRecording,
                         onStopRecording,
                         onResetRecorder,
                         onSendRecording,
                         onPlayTranslatedText
                     }: ControlDockProps) => {
    const renderFirstButtonIcon = () => {
        if (state === "recording") return <Square className="w-6 h-6"/>;
        else if (state === "ready") return <Trash className="w-6 h-6"/>;
        else return <Video className="w-6 h-6"/>;
    };

    const handleFirstButtonClick = () => {
        if (state === "recording") onStopRecording();
        else if (state === "ready") onResetRecorder();
        else onStartRecording();
    };

    return (
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
                    aria-label={state === "recording" ? "Stop recording" : state === "ready" ? "Delete recording" : "Start recording"}
                >
                    {renderFirstButtonIcon()}
                </button>

                {/* Play Button */}
                <button
                    onClick={onPlayTranslatedText}
                    disabled={!isTextAvailable || ["recording", "loading"].includes(state)}
                    className={cn(
                        "p-3 rounded-full transition-all transform hover:scale-110",
                        isPlaying ? "bg-green-900 text-white shadow-md" : "bg-gray-100 hover:bg-gray-200",
                        (!isTextAvailable || ["recording", "loading"].includes(state)) && "opacity-50 cursor-not-allowed",
                    )}
                    aria-label={isPlaying ? "Restart speech" : "Play translated text"}
                >
                    <Play className="w-6 h-6"/>
                </button>

                {/* Send Button */}
                <button
                    onClick={state === "ready" ? onSendRecording : onResetRecorder}
                    disabled={["recording", "loading", "default"].includes(state)}
                    className={cn(
                        "p-3 rounded-full transition-all transform hover:scale-110",
                        state === "ready"
                            ? "bg-green-900 text-white shadow-md"
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
    );
};

export default ControlDock;