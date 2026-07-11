Output demo.gif
Set FontSize 24
Set Width 1600
Set Height 992
Set Padding 38
Set TypingSpeed 35ms
Set WaitTimeout 300s
Set WaitPattern /^\$\s*$/
Set Theme { "name": "droste-ink", "background": "#131313", "foreground": "#f3f2ee", "cursor": "#8fa2ff", "black": "#131313", "red": "#e06c6c", "green": "#57c07d", "yellow": "#d9b455", "blue": "#8fa2ff", "magenta": "#c580d6", "cyan": "#5cc4ca", "white": "#f3f2ee", "brightBlack": "#8a897f", "brightRed": "#f08a8a", "brightGreen": "#79d49c", "brightYellow": "#e8cb76", "brightBlue": "#aab8ff", "brightMagenta": "#d69fe3", "brightCyan": "#7fd6db", "brightWhite": "#ffffff", "selection": "#33322e" }

Hide
Type "source ./env.sh && cd ./rec && export PS1='\[\e[38;5;111m\]$\[\e[0m\] ' && clear"
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
