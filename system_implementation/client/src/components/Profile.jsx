import {
    Avatar,
    AvatarFallback,
    AvatarImage,
} from "@/components/ui/avatar"

export default function Profile() {
    return (
        <Avatar>
            <AvatarImage src="https://github.com/shadcn.png" alt="@snoopy" />
            <AvatarFallback>CN</AvatarFallback>
        </Avatar>
    )
}
