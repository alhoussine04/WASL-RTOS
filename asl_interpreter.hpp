#ifndef ASL_INTERPRETER_HPP
#define ASL_INTERPRETER_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <nlohmann/json.hpp> // For JSON parsing

#include <string>
#include <vector>
#include <deque>
#include <map>
#include <thread>
#include <mutex>
#include <atomic>
#include <optional>
#include <chrono>

class ASLInterpreter {
public:
    ASLInterpreter(
        const std::string& modelXml = "asl-recognition-0004.xml",
        const std::string& modelBin = "asl-recognition-0004.bin",
        const std::string& classesFile = "MSASL_classes.json",
        const std::string& device = "CPU",
        double confidenceThreshold = 0.80,
        int stabilityThreshold = 3,
        int cooldownThreshold = 45, // Frames
        double motionThreshold = 50.0);

    ~ASLInterpreter(); // Destructor to clean up thread

    // Process a single frame and return the visualized frame
    cv::Mat processFrame(const cv::Mat& frame);

    // Start/Stop the background inference thread
    void startInferenceThread();
    void stopInferenceThread();

    // Save the transcript
    void saveTranscript(const std::string& filename = "asl_transcript.txt");

    // Get the current transcript
    std::vector<std::string> getTranscript() const;


private:
    // --- Configuration ---
    double confidenceThreshold_;
    int stabilityThreshold_;
    int cooldownThreshold_;
    double motionThreshold_;
    int clipLength_;
    int height_;
    int width_;

    // --- OpenVINO ---
    ov::Core core_;
    ov::CompiledModel compiledModel_;
    ov::InferRequest inferRequest_;
    ov::Output<const ov::Node> inputNode_;
    ov::Output<const ov::Node> outputNode_;
    std::vector<std::string> classes_;

    // --- Buffers ---
    std::deque<cv::Mat> frameBuffer_;
    std::deque<std::string> predictionBuffer_;

    // --- State Variables ---
    std::vector<std::string> transcript_;
    std::string lastPrediction_;
    int predictionCooldown_;
    std::map<std::string, int> stabilityCounter_; // Counts consecutive frames for a stable prediction

    // --- Motion Detection ---
    cv::Mat prevGray_;
    int noMotionFrames_;

    // --- Threading ---
    std::thread inferenceThread_;
    std::atomic<bool> inferenceRunning_;
    std::optional<ov::Tensor> latestResult_; // Optional to handle case before first inference
    std::mutex resultMutex_;

    // --- Helper Methods ---
    void loadAslModel(const std::string& modelXml, const std::string& modelBin, const std::string& device);
    void loadClasses(const std::string& classesFile);
    ov::Tensor processFrames(const std::deque<cv::Mat>& frames); // Prepare frames for inference
    void runInferenceThread(); // The actual function run by the thread
};

#endif // ASL_INTERPRETER_HPP
