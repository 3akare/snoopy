export type RecorderState = "default" | "recording" | "ready" | "loading" | "error" | "show-text";

export interface RecorderProps {
    onReset: () => void;
    onRetry: () => void;
    state: RecorderState;
}
