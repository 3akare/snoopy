import { RecorderProps } from "@/app/types/RecorderProps";
import { Loader2 } from "lucide-react";
import { forwardRef } from "react";

const Recorder = forwardRef<HTMLDivElement, RecorderProps>(({ onReset, onRetry, state }, ref) => {
  return (
    <div ref={ref} className="flex flex-col items-center justify-center h-full">
      {state === "default" && (
        <div className="flex flex-col items-center justify-center h-full">
          <p className="text-xl text-center text-white">Click the video button to begin</p>
        </div>
      )}
      {state === "recording" && (
        <div className="flex flex-col items-center justify-center h-full">
          <div className="text-center">
            <div className="inline-flex items-center gap-2 bg-green-900/30 text-green-400 px-4 py-2 rounded-full">
              <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
              <p className="font-medium">Recording...</p>
            </div>
          </div>
        </div>
      )}
      {state === "ready" && (
        <div className="flex flex-col items-center justify-center h-full">
          <p className="text-xl text-center mb-4 text-white">Ready to send</p>
          <p className="text-base text-center text-green-400">
            Click Send to upload the video or Delete to discard the recording.
          </p>
        </div>
      )}
      {state === "loading" && (
        <div className="flex flex-col items-center justify-center h-full">
          <div className="text-center">
            <Loader2 className="w-10 h-10 text-green-500 animate-spin mx-auto mb-4" />
            <p className="text-lg text-white">Processing your recording...</p>
          </div>
        </div>
      )}
      {state === "error" && (
        <div className="flex flex-col items-center justify-center h-full">
          <div className="text-center">
            <p className="text-red-400 text-xl mb-2">An error occurred</p>
            <div className="flex gap-4 justify-center">
              <button
                onClick={onRetry}
                className="px-4 py-2 bg-green-400/30 text-green-400 rounded-md text-sm hover:bg-green-400/50"
              > Retry
              </button>
              <button
                onClick={onReset}
                className="px-4 py-2 bg-red-900/30 text-red-400 rounded-md text-sm hover:bg-red-900/50"
              > Delete
              </button>
            </div>
          </div>
        </div>
      )
      }
    </div>
  );
});

export default Recorder;
