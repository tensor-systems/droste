Output demo.gif
Set FontSize 15
Set Width 1000
Set Height 620
Set Padding 24
Set TypingSpeed 35ms
Set WaitTimeout 300s
Set WaitPattern /^\$\s*$/
Set Theme { "name": "droste-paper", "background": "#f3f2ee", "foreground": "#131313", "cursor": "#1f3cc4", "black": "#131313", "red": "#b02a2a", "green": "#1a6e3c", "yellow": "#8a6d00", "blue": "#1f3cc4", "magenta": "#7b2d8b", "cyan": "#0f6f74", "white": "#f3f2ee", "brightBlack": "#66655f", "brightRed": "#b02a2a", "brightGreen": "#1a6e3c", "brightYellow": "#8a6d00", "brightBlue": "#1f3cc4", "brightMagenta": "#7b2d8b", "brightCyan": "#0f6f74", "brightWhite": "#ffffff", "selection": "#d9d7cf" }

Hide
Type "source ./env.sh && cd ./rec && export PS1='$ ' && clear"
Enter
Sleep 1s
Show

Type "ls -lh"
Enter
Sleep 2s

Type@40ms `droste "Which customer had a failed charge, and why? How many timeouts, blaming which upstream?" server.log --verbose`
Enter
Wait@300s
Sleep 4s
