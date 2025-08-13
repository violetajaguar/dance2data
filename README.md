# dance2data

Turn movement into data and data into movement.
This repo hosts experiments that map dance, gesture, and camera input to real-time visuals, sound, and AI responses.

## Quickstart
```bash
# create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt

# run your main script
python app.py
```

## What’s inside
- app.py or src/ — main loop for capture, analysis, and output
- data/ — raw and processed inputs (ignored by Git)
- models/ — ML models and checkpoints (ignored)
- outputs/ — renders, logs, previews (ignored)

## Reproducibility
```bash
python3 -m pip freeze > requirements.txt
```

## Notes
- .venv, caches, and big binaries are ignored
- Private media should go in data/ and be excluded from Git

## License
MIT 
