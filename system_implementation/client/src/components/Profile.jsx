import {
    Avatar,
    AvatarFallback,
    AvatarImage,
} from "@/components/ui/avatar"

export default function Profile({src, alt, fallback}) {
    return (
        <Avatar>
            <AvatarImage src={src} alt={alt} />
            <AvatarFallback className={"bg-emerald-600 text-white"}>{fallback}</AvatarFallback>
        </Avatar>
    )
}
