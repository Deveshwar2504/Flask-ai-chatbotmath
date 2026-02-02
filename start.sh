#!/usr/bin/env bash
export FLASK_APP=chatbotflask.py
gunicorn chatbotflask:app
chmod +x start.sh



