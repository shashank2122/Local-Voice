# Local Voice 🎤

![Local Voice](https://img.shields.io/badge/Local%20Voice-Real%20Time%20Voice%20Assistant-blue)

Welcome to **Local Voice**, a real-time, offline voice assistant designed for Linux and Raspberry Pi. This project leverages local large language models (LLMs) through Ollama, speech-to-text capabilities with Vosk, and text-to-speech features using Piper. Enjoy fast, wake-free voice interaction without relying on the cloud or APIs. Just Python, a microphone, and your voice.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)
- [Releases](#releases)

## Features

- **Offline Functionality**: Operate without an internet connection.
- **Real-Time Interaction**: Respond to your commands instantly.
- **Local LLMs**: Utilize powerful language models locally.
- **Speech Recognition**: Convert spoken words into text accurately.
- **Text-to-Speech**: Generate human-like speech from text.
- **Compatibility**: Works seamlessly on Linux and Raspberry Pi.

## Installation

To get started with Local Voice, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shashank2122/Local-Voice.git
   cd Local-Voice
   ```

2. **Install Dependencies**:
   Make sure you have Python installed. You can install the required packages using:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Ollama**:
   Follow the instructions on the [Ollama website](https://ollama.com) to install and configure Ollama.

4. **Set Up Vosk**:
   Install Vosk for speech recognition:
   ```bash
   sudo apt-get install vosk-api
   ```

5. **Set Up Piper**:
   Install Piper for text-to-speech:
   ```bash
   sudo apt-get install piper
   ```

6. **Configure Microphone**:
   Ensure your microphone is connected and properly configured in your system settings.

## Usage

To run the Local Voice assistant, execute the following command in your terminal:

```bash
python main.py
```

Once the assistant is running, simply speak your command, and it will respond accordingly. 

### Example Commands

- "What’s the weather today?"
- "Play my favorite song."
- "Set a timer for 10 minutes."

## Technologies Used

- **Python**: The primary programming language for this project.
- **Ollama**: For running local LLMs.
- **Vosk**: For speech-to-text conversion.
- **Piper**: For text-to-speech capabilities.
- **Linux**: The operating system on which this project runs.
- **Raspberry Pi**: The hardware platform for this project.

## Contributing

We welcome contributions from everyone! If you want to help improve Local Voice, please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. Make your changes and commit them:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/YourFeature
   ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, feel free to reach out:

- **Email**: your-email@example.com
- **GitHub**: [shashank2122](https://github.com/shashank2122)

## Releases

To download the latest version of Local Voice, visit the [Releases section](https://github.com/shashank2122/Local-Voice/releases). Here, you can find the latest updates and files you need to execute.

For any issues or feature requests, please check the "Releases" section as well.

---

Thank you for your interest in Local Voice! We hope you enjoy using this offline voice assistant as much as we enjoyed building it. Your feedback is valuable to us. Happy coding!