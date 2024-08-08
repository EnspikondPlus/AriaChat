# AriaChat
A rudimentary chatbot with speech and text input/output. A minimalistic UI was made using PyGame to communicate with Aria, and ML models used are from the ```transformers``` package. Models setup are Pygmalion-1.3b and DialoGPT-Large.
![idle](https://github.com/user-attachments/assets/ee18cc14-199f-455b-976e-776bc7611acc)
## Installation
All code and processing is run locally, but an internet connection is needed to install datasets for the language and speech models when the program is initally run. Clone the repository, install necessary dependencies, and run either python script.
## Usage
Communicate with Aria by pressing "Record" on the GUI, and "Stop" to finish your recording. Press "Talk" to process the recording and get a response from Aria. A transcript of your recording and Aria's response will be shown in the textbox, which you can edit before your next recording. Aria may reply based on information within the transcript.
Aria will animate as she responds verbally. You can exit the program by clicking the "Exit" button.
## Troubleshooting
### Installation Problems
Check you have the correct dependencies or compatible version of python to support all the packages.
### Aria Isn't Responding Properly
Uhh... not much we can do about that. The models used are very small and designed to run relatively quickly, and locally, on a PC or laptop. If you have the processing power, you can search on [huggingface](https://huggingface.co/models) for a better model and substitute that into the code.
