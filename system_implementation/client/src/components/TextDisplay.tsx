import {TextDisplayProps} from "@/types/TextDisplayProps.ts";
import {forwardRef} from "react";
import {Copy} from "lucide-react";
import {cn} from "@/lib/utils";

const TextDisplay = forwardRef<HTMLDivElement, TextDisplayProps>(({
                                                                      translatedText,
                                                                      highlightedIndex,
                                                                      words,
                                                                      onCopyToClipboard,
                                                                  }, ref) => {
    if (!translatedText) return null;
    return (
        <div ref={ref} className="relative flex-1 p-6 overflow-auto">
            <div className="text-3xl text-center leading-relaxed md:text-left md:text-2xl ">
                {words.map((word, index) => (
                    <span key={index}
                          className={cn(
                              "transition-colors duration-200",
                              index === highlightedIndex ? "text-gray-900 font-medium" : "text-gray-400",
                          )}>
                        {word}{" "}
                    </span>
                ))}
            </div>
            <button
                onClick={onCopyToClipboard}
                className="absolute top-3 right-3 p-2 rounded-md hover:bg-gray-100 cursor-pointer"
                aria-label="Copy to clipboard"
            >
                <Copy className="w-5 h-5"/>
            </button>
        </div>
    );
});

export default TextDisplay;