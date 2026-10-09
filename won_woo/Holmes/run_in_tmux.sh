#!/bin/bash

# Define session name
SESSION_NAME="holmes_session"

# Check if session already exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
  # Create a new detached tmux session
  tmux new-session -d -s $SESSION_NAME

  # Send commands to the tmux session
  tmux send-keys -t $SESSION_NAME "cd /home/viglab2/won_woo/Holmes" C-m
  tmux send-keys -t $SESSION_NAME "conda activate holmes" C-m
  tmux send-keys -t $SESSION_NAME "python utils/adaptive_rewriter.py" C-m

  echo "Tmux session '$SESSION_NAME' created and script started."
  echo "To attach to this session, run: tmux attach -t $SESSION_NAME"
else
  echo "Tmux session '$SESSION_NAME' already exists."
  echo "To attach, run: tmux attach -t $SESSION_NAME"
fi
