#include "asl_interpreter.hpp"
#include <iostream>
#include <string>
#include <vector>
#include <deque>
#include <numeric> // std::accumulate
#include <chrono>

std::string getCurrentTimestamp();
	
// Basic command-line argument parsing (replace with a library like cxxopts for robustness)
struct Args {
    std::string modelXml = "asl-recognition-0004.xml";
    std::string modelBin = "asl-recognition-0004.bin";
    std::string classes = "MSASL_classes.json";
    std::string device = "CPU";
    double confidence = 0.80;
    int stability = 3;
    double motion = 50.0; // Changed type to double to match constructor
    int camera = 0;
    std::string output = "asl_transcript.txt";
    bool record = false;
    std::string recordFile = ""; // Set if record is true
};

Args parseArgs(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model-xml" && i + 1 < argc) args.modelXml = argv[++i];
        else if (arg == "--model-bin" && i + 1 < argc) args.modelBin = argv[++i];
        else if (arg == "--classes" && i + 1 < argc) args.classes = argv[++i];
        else if (arg == "--device" && i + 1 < argc) args.device = argv[++i];
        else if (arg == "--confidence" && i + 1 < argc) args.confidence = std::stod(argv[++i]);
        else if (arg == "--stability" && i + 1 < argc) args.stability = std::stoi(argv[++i]);
        else if (arg == "--motion" && i + 1 < argc) args.motion = std::stod(argv[++i]); // Use stod
        else if (arg == "--camera" && i + 1 < argc) args.camera = std::stoi(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) args.output = argv[++i];
        else if (arg == "--record") args.record = true;
        else std::cerr << "Warning: Ignoring unknown or incomplete argument: " << arg << std::endl;
    }
    if (args.record) {
         std::string timestamp = getCurrentTimestamp(); // Assumes getCurrentTimestamp is available
         args.recordFile = "asl_recording_" + timestamp + ".avi";
         std::cout << "Recording enabled. Output file: " << args.recordFile << std::endl;
    }
    return args;
}


int main(int argc, char* argv[]) {
    Args args = parseArgs(argc, argv);

    // --- Initialize Video Capture ---
    cv::VideoCapture cap;
    if (!cap.open(args.camera)) {
        std::cerr << "Error: Could not open camera index " << args.camera << std::endl;
        return -1;
    }

    // --- Set Camera Properties ---
    int frameWidth = 640;
    int frameHeight = 480;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, frameWidth);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frameHeight);
    // cap.set(cv::CAP_PROP_FPS, 30); // Setting FPS might not always work reliably

    // --- Initialize ASL Interpreter ---
    std::unique_ptr<ASLInterpreter> interpreter;
    try {
        interpreter = std::make_unique<ASLInterpreter>(
            args.modelXml,
            args.modelBin,
            args.classes,
            args.device,
            args.confidence,
            args.stability,
            45, // Cooldown hardcoded as it wasn't in args parser (add if needed)
            args.motion
        );
    } catch (const std::exception& e) {
         std::cerr << "Failed to initialize ASL Interpreter: " << e.what() << std::endl;
         cap.release();
         return -1;
    }


    // --- Start Inference Thread ---
    interpreter->startInferenceThread();

    // --- Initialize Video Writer (if recording) ---
    cv::VideoWriter videoWriter;
    if (args.record) {
        // Use MJPG or DIVX/XVID which are common. Check available codecs on your system.
        int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
        // Use actual camera FPS if possible, otherwise default to 20.0
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 20.0; // Default if couldn't get FPS
        if (!videoWriter.open(args.recordFile, fourcc, fps, cv::Size(frameWidth, frameHeight))) {
            std::cerr << "Error: Could not open video writer for " << args.recordFile << std::endl;
            args.record = false; // Disable recording if failed
        }
    }

    // --- FPS Calculation ---
    std::deque<double> fpsBuffer;
    const size_t fpsBufferSize = 30;
    auto lastTime = std::chrono::high_resolution_clock::now();
    double avgFps = 0.0;

    std::cout << "ASL Interpreter Started - Press 'q' to quit, 's' to save transcript" << std::endl;

    cv::Mat frame;
    while (cap.isOpened()) {
        if (!cap.read(frame)) {
            std::cerr << "Error: Can't receive frame (stream end?). Exiting..." << std::endl;
            break;
        }
        if (frame.empty()) {
             std::cerr << "Warning: Received empty frame." << std::endl;
             continue;
        }


        // --- Calculate FPS ---
        auto currentTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = currentTime - lastTime;
        lastTime = currentTime;

        if (elapsed.count() > 0) {
            double currentFps = 1.0 / elapsed.count();
            fpsBuffer.push_back(currentFps);
            if (fpsBuffer.size() > fpsBufferSize) {
                fpsBuffer.pop_front();
            }
            avgFps = std::accumulate(fpsBuffer.begin(), fpsBuffer.end(), 0.0) / fpsBuffer.size();
        }

        // --- Process Frame ---
        cv::Mat visFrame = interpreter->processFrame(frame);

        // --- Display FPS ---
        std::ostringstream fpsOss;
        fpsOss << "FPS: " << std::fixed << std::setprecision(1) << avgFps;
        cv::putText(visFrame, fpsOss.str(), cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // --- Display Help ---
        cv::putText(visFrame, "Press 'q' to quit, 's' to save transcript",
                    cv::Point(10, visFrame.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

        // --- Show Frame ---
        cv::imshow("ASL Interpreter", visFrame);

        // --- Record Frame ---
        if (args.record && videoWriter.isOpened()) {
            videoWriter.write(visFrame);
        }

        // --- Handle Key Presses ---
        int key = cv::waitKey(1); // Wait 1ms for a key press
        if (key != -1) {
            key &= 0xFF; // Mask for ASCII value
            if (key == 'q' || key == 27) { // 'q' or ESC key
                std::cout << "Quit key pressed. Exiting..." << std::endl;
                break;
            } else if (key == 's') {
                 std::cout << "Save key pressed." << std::endl;
                interpreter->saveTranscript(args.output);
            }
        }
    } // End while loop


    // --- Cleanup ---
    std::cout << "Exiting main loop. Cleaning up..." << std::endl;
    interpreter->stopInferenceThread(); // Stop thread before saving final transcript

    std::cout << "Saving final transcript..." << std::endl;
    interpreter->saveTranscript(args.output); // Save final transcript

    if (videoWriter.isOpened()) {
        std::cout << "Releasing video writer..." << std::endl;
        videoWriter.release();
    }
     std::cout << "Releasing camera capture..." << std::endl;
    cap.release();
     std::cout << "Destroying OpenCV windows..." << std::endl;
    cv::destroyAllWindows();

    // --- Print Final Transcript ---
    std::cout << "\nFinal Transcript:" << std::endl;
    std::vector<std::string> finalTranscript = interpreter->getTranscript();
    for (size_t i = 0; i < finalTranscript.size(); ++i) {
        std::cout << finalTranscript[i] << (i == finalTranscript.size() - 1 ? "" : " ");
    }
    std::cout << std::endl;

    std::cout << "Cleanup complete. Exiting program." << std::endl;
    return 0;
}
