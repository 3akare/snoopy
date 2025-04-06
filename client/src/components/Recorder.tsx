import {RecorderProps} from "@/types/RecorderProps.ts";
import {Loader2} from "lucide-react";
import {forwardRef} from "react";

const Recorder = forwardRef<HTMLDivElement, RecorderProps>(({onReset, state,}, ref) => {
    return (
        <div ref={ref} className="flex flex-col items-center justify-center h-full">
            {state === "default" && (
                <p className="text-xl text-center flex items-center align-items">
                    Click Video to get started.
                </p>
            )}
            {state === "recording" && (
                <div className="text-center">
                    <div className="inline-flex items-center gap-2 bg-red-100 text-red-600 px-4 py-2 rounded-full">
                        <div className="w-3 h-3 bg-red-600 rounded-full animate-pulse"></div>
                        <p className="text-xl">Recording...</p>
                    </div>
                </div>
            )}
            {state === "ready" && (
                <p className="text-xl text-center mb-4">
                    Click Send to upload the video or Delete to discard the recording.
                </p>
            )}
            {state === "loading" && (
                <div className="text-center">
                    <Loader2 className="w-14 h-14 text-green-900 animate-spin mx-auto mb-4"/>
                </div>
            )}
            {state === "error" && (
                <div className="text-center">
                    <p className="text-red-500 text-xl mb-2">An error occurred</p>
                    {/* Error message will be passed from the parent */}
                    <div className="flex gap-2 justify-center">
                        <button onClick={onReset} className="px-4 py-2 bg-red-100 text-red-600 rounded-md text-sm">
                            Delete
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
});

export default Recorder;