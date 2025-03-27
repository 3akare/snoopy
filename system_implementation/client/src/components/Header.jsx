import avatar from "/avatar.jpeg";

export default function Header() {
    return (
        <header className={"container mx-auto max-w-5xl"}>
            <nav className={"w-full flex items-center justify-between h-12 py-12"}>
                <h1 className={"text-2xl font-semibold"}> Snoopy </h1>
                <img src={avatar} alt={"avatar"} className={"size-12 rounded-full bg-white mx-4"}></img>
            </nav>
        </header>
    );
}