#include "asl_interpreter.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <ctime>

// Helper to get timestamp string
std::string getCurrentTimestamp() {
    std::time_t now = std::time(nullptr);
    char buf[20]; // Sufficient buffer size
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", std::localtime(&now));
    return std::string(buf);
}

ASLInterpreter::ASLInterpreter(
    const std::string& modelXml,
    const std::string& modelBin,
    const std::string& classesFile,
    const std::string& device,
    double confidenceThreshold,
    int stabilityThreshold,
    int cooldownThreshold,
    double motionThreshold)
    : confidenceThreshold_(confidenceThreshold),
      stabilityThreshold_(stabilityThreshold),
      cooldownThreshold_(cooldownThreshold),
      motionThreshold_(motionThreshold),
      clipLength_(0), // Will be set after model load
      height_(0), // Will be set after model load
      width_(0), // Will be set after model load
      predictionCooldown_(0),
      noMotionFrames_(0),
      inferenceRunning_(false)
{
    try {
        loadClasses(classesFile);
        loadAslModel(modelXml, modelBin, device); // Loads model and sets shape members
        // Initialize buffer sizes based on model input
        frameBuffer_ = std::deque<cv::Mat>(clipLength_); // Initialize with default cv::Mat
        predictionBuffer_ = std::deque<std::string>(8); // Fixed size like Python code
    } catch (const std::exception& e) {
        std::cerr << "Error during ASLInterpreter initialization: " << e.what() << std::endl;
        throw;
    }
}

ASLInterpreter::~ASLInterpreter() {
    stopInferenceThread(); // Ensure thread is stopped and joined
}

void ASLInterpreter::loadClasses(const std::string& classesFile) {
    if (!std::filesystem::exists(classesFile)) {
        throw std::runtime_error("Classes file not found: " + classesFile);
    }

    std::ifstream ifs(classesFile);
    if (!ifs.is_open()) {
        throw std::runtime_error("Could not open classes file: " + classesFile);
    }

    try {
        nlohmann::json j;
        ifs >> j;
        classes_ = j.get<std::vector<std::string>>();
        std::cout << "Loaded " << classes_.size() << " classes." << std::endl;
    } catch (const nlohmann::json::exception& e) {
        throw std::runtime_error("Error parsing classes JSON file: " + std::string(e.what()));
    }
}

void ASLInterpreter::loadAslModel(const std::string& modelXml, const std::string& modelBin, const std::string& device) {
    if (!std::filesystem::exists(modelXml) || !std::filesystem::exists(modelBin)) {
        throw std::runtime_error("Model files not found: " + modelXml + " or " + modelBin);
    }

    try {
        auto model = core_.read_model(modelXml, modelBin);
        compiledModel_ = core_.compile_model(model, device);
        inferRequest_ = compiledModel_.create_infer_request(); // Create one request
        inputNode_ = compiledModel_.input();
        outputNode_ = compiledModel_.output();
        // Get input shape: [1, 3, clip_length, height, width]
        ov::Shape inputShape = inputNode_.get_shape();
        if (inputShape.size() != 5) {
            throw std::runtime_error("Unexpected model input shape dimensions: " + std::to_string(inputShape.size()));
        }

        clipLength_ = static_cast<int>(inputShape[2]);
        height_ = static_cast<int>(inputShape[3]);
        width_ = static_cast<int>(inputShape[4]);
        std::cout << "Model loaded successfully on " << device << std::endl;
        std::cout << "Input Shape: [1, 3, " << clipLength_ << ", " << height_ << ", " << width_ << "]" << std::endl;
    } catch (const ov::Exception& e) {
        throw std::runtime_error("OpenVINO Error loading model: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error loading model: " + std::string(e.what()));
    }
}

ov::Tensor ASLInterpreter::processFrames(const std::deque<cv::Mat>& frames) {
    // FIXED: Explicitly use f32 element type instead of getting from inputNode_
    ov::Tensor inputTensor(ov::element::f32, inputNode_.get_shape());
    // FIXED: Explicitly cast to float pointer to avoid type mismatch
    float* data = inputTensor.data<float>();

    if (frames.size() != clipLength_) {
        throw std::runtime_error("Incorrect number of frames for processing: " +
            std::to_string(frames.size()) + ", expected " +
            std::to_string(clipLength_));
    }

    // Layout is [1, 3, clip_length, height, width]
    for (int n = 0; n < clipLength_; ++n) {
        cv::Mat resizedFrame;
        // Check if frame is valid before processing
        if (frames[n].empty()) {
            throw std::runtime_error("Empty frame encountered in buffer at index " + std::to_string(n));
        }

        cv::resize(frames[n], resizedFrame, cv::Size(width_, height_));
        cv::Mat rgbFrame;
        cv::cvtColor(resizedFrame, rgbFrame, cv::COLOR_BGR2RGB); // Model expects RGB
        // FIXED: Convert to float and normalize to [0,1] range
        cv::Mat floatFrame;
        rgbFrame.convertTo(floatFrame, CV_32F, 1.0/255.0);
        // Copy frame data to the tensor buffer with layout C, N, H, W
        for (int c = 0; c < 3; ++c) { // Channels (RGB)
            for (int h = 0; h < height_; ++h) { // Height
                for (int w = 0; w < width_; ++w) { // Width
                    size_t offset = c * clipLength_ * height_ * width_ +
                                   n * height_ * width_ +
                                   h * width_ +
                                   w;
                    // FIXED: Access pixel data correctly for floating point values
                    data[offset] = floatFrame.at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }

    return inputTensor;
}

void ASLInterpreter::runInferenceThread() {
    while (inferenceRunning_) {
        std::deque<cv::Mat> currentFrames;
        bool bufferReady = false;
        {
            if (frameBuffer_.size() == clipLength_) {
                bool allValid = true;
                for(const auto& frm : frameBuffer_) {
                    if (frm.empty()) {
                        allValid = false;
                        break;
                    }
                }
                
                if (allValid) {
                    currentFrames = frameBuffer_; // Copy constructor performs deep copy for cv::Mat
                    bufferReady = true;
                }
            }
        }

        if (bufferReady && !currentFrames.empty()) {
            try {
                // 1. Process frames (convert to tensor)
                ov::Tensor inputTensor = processFrames(currentFrames);
                // 2. Set input tensor for the request
                inferRequest_.set_tensor(inputNode_, inputTensor);
                // 3. Run inference
                inferRequest_.infer();
                // 4. Get output tensor
                ov::Tensor outputTensor = inferRequest_.get_tensor(outputNode_);
                // 5. Update latest result (thread-safe)
                // FIXED: Simplified tensor handling to avoid type mismatch
                std::lock_guard<std::mutex> lock(resultMutex_);
                latestResult_ = outputTensor; // Simple assignment
            } catch (const ov::Exception& e) {
                std::cerr << "OpenVINO Inference Error: " << e.what() << std::endl;
                
                std::lock_guard<std::mutex> lock(resultMutex_);
                latestResult_.reset();
                
            } catch (const std::exception& e) {
                std::cerr << "Inference Thread Error: " << e.what() << std::endl;
                
                std::lock_guard<std::mutex> lock(resultMutex_);
                latestResult_.reset();
            }
        }
        
        // Sleep briefly to avoid busy-waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    std::cout << "Inference thread finished." << std::endl;
}

void ASLInterpreter::startInferenceThread() {
    if (!inferenceRunning_) {
        inferenceRunning_ = true;
        if (inferenceThread_.joinable()) {
            inferenceThread_.join();
        }
        
        inferenceThread_ = std::thread(&ASLInterpreter::runInferenceThread, this);
        std::cout << "Inference thread started." << std::endl;
    }
}

void ASLInterpreter::stopInferenceThread() {
    if (inferenceRunning_) {
        inferenceRunning_ = false; // Signal thread to stop
        if (inferenceThread_.joinable()) {
            try {
                inferenceThread_.join(); // Wait for thread to finish
                std::cout << "Inference thread successfully joined." << std::endl;
            } catch (const std::system_error& e) {
                std::cerr << "Error joining inference thread: " << e.what() << std::endl;
            }
        }
    }
}

cv::Mat ASLInterpreter::processFrame(const cv::Mat& frame) {
    cv::Mat visFrame = frame.clone(); // Create a copy for visualization
    float currentConfidence = 0.0f;  // Add this declaration
    
    // Declare these variables at function scope to fix the scope issues
    const float* outputData = nullptr;
    size_t numClasses = 0;
    
    // --- Motion Detection ---
    cv::Mat gray, blurredGray, frameDiff, thresh;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurredGray, cv::Size(21, 21), 0);
    
    double motionScore = 0.0;
    if (!prevGray_.empty()) {
        cv::absdiff(prevGray_, blurredGray, frameDiff);
        cv::threshold(frameDiff, thresh, 25, 255, cv::THRESH_BINARY);
        motionScore = cv::sum(thresh)[0] / 255.0; // Sum of non-zero pixels
    }
    
    prevGray_ = blurredGray.clone(); // Update previous frame for next iteration
    
    // --- Frame Buffering ---
    frameBuffer_.push_back(frame.clone()); // Add deep copy
    while(frameBuffer_.size() > clipLength_) {
        frameBuffer_.pop_front();
    }
    
    // --- Prediction Logic ---
    std::string currentPredictionText = "Initializing...";
    std::string stabilityText = "";
    
    if (motionScore < motionThreshold_) {
        noMotionFrames_++;
        if (noMotionFrames_ > 10) { // Reset stability if no motion for a while
            stabilityCounter_.clear();
            predictionBuffer_ = std::deque<std::string>(predictionBuffer_.size()); // Clear prediction buffer
        }
        
        if (noMotionFrames_ > 20) {
            currentPredictionText = "No motion detected";
        } else {
            currentPredictionText = "Low motion";
        }
        
    } else {
        noMotionFrames_ = 0; // Reset no motion counter
        // Get latest result from inference thread (thread-safe)
        std::optional<ov::Tensor> currentResult;
        
        {
            std::lock_guard<std::mutex> lock(resultMutex_);
            if (latestResult_) {
                // FIXED: Use simple assignment instead of creating a new tensor
                currentResult = latestResult_;
            }
        }
        
        if (currentResult) {
            // Explicitly specify data type for output data access
            outputData = currentResult->data<float>();
            numClasses = currentResult->get_size();
            
            // Inside your processFrame method where you currently have the warning
            if (numClasses != classes_.size()) {
                // Instead of just showing a warning and stopping prediction
                std::cout << "Note: Output tensor size (" << numClasses
                          << ") is smaller than number of classes (" << classes_.size()
                          << "). Using only the first " << numClasses << " classes." << std::endl;
                
                // Continue processing using only the first numClasses entries
                if (numClasses < classes_.size()) {
                    // Find the prediction with highest confidence
                    float maxConfidence = 0.0f;
                    int maxConfidenceIdx = -1;
                    for (size_t i = 0; i < numClasses; ++i) {
                        float confidence = outputData[i];
                        if (confidence > maxConfidence) {
                            maxConfidence = confidence;
                            maxConfidenceIdx = i;
                        }
                    }
                    
                    // Only show prediction if confidence exceeds threshold
                    if (maxConfidence >= confidenceThreshold_) {
                        currentPredictionText = classes_[maxConfidenceIdx];
                        currentConfidence = maxConfidence;
                    } else {
                        currentPredictionText = ""; // No confident prediction
                        currentConfidence = 0.0f;
                    }
                    
                } else {
                    // Handle the unlikely case where numClasses > classes_.size()
                    std::cerr << "Error: Output tensor size exceeds available classes" << std::endl;
                    currentPredictionText = "Output size error";
                    currentConfidence = 0.0f;
                }
                
            } else {
                // Your original code for when sizes match
                // Find prediction with highest confidence
                float maxConfidence = 0.0f;
                int maxConfidenceIdx = -1;
                for (size_t i = 0; i < numClasses; ++i) {
                    float confidence = outputData[i];
                    if (confidence > maxConfidence) {
                        maxConfidence = confidence;
                        maxConfidenceIdx = i;
                    }
                }
                
                // Only show prediction if confidence exceeds threshold
                if (maxConfidence >= confidenceThreshold_) {
                    currentPredictionText = classes_[maxConfidenceIdx];
                    currentConfidence = maxConfidence;
                } else {
                    currentPredictionText = ""; // No confident prediction
                    currentConfidence = 0.0f;
                }
            }
            
        } else {
            currentPredictionText = "No inference result yet";
            
            // This else block had the issues with undeclared variables
            // Now that outputData and numClasses are declared at function scope,
            // we need to check if they're valid before using them
            if (outputData != nullptr && numClasses > 0) {
                // Find top prediction
                int maxIndex = -1;
                float maxConf = -1.0f;
                for (size_t i = 0; i < numClasses; ++i) {
                    if (outputData[i] > maxConf) {
                        maxConf = outputData[i];
                        maxIndex = static_cast<int>(i);
                    }
                }
                
                if (maxIndex >= 0 && maxIndex < classes_.size() && maxConf >= confidenceThreshold_) {
                    std::string prediction = classes_[maxIndex];
                    float confidence = maxConf;
                    // Add prediction to buffer
                    predictionBuffer_.push_back(prediction);
                    while(predictionBuffer_.size() > 8) { // Maintain maxlen
                        predictionBuffer_.pop_front();
                    }
                    
                    // Update stability counter
                    if (stabilityCounter_.count(prediction)) {
                        stabilityCounter_[prediction]++;
                    } else {
                        // Clear counter if prediction changes
                        stabilityCounter_.clear();
                        stabilityCounter_[prediction] = 1;
                    }
                    
                    // Check for stable prediction
                    std::string stablePrediction = "";
                    if (stabilityCounter_[prediction] >= stabilityThreshold_) {
                        // Check how often it appears in the recent buffer
                        int bufferCount = 0;
                        for(const auto& p : predictionBuffer_) {
                            if (p == prediction) {
                                bufferCount++;
                            }
                        }
                        
                        if (bufferCount >= predictionBuffer_.size() / 2) {
                            stablePrediction = prediction;
                        }
                    }
                    
                    // Add to transcript if stable and meets cooldown criteria
                    if (!stablePrediction.empty() &&
                        (stablePrediction != lastPrediction_ || predictionCooldown_ >= cooldownThreshold_)) {
                        std::ostringstream oss;
                        oss << stablePrediction << " (" << std::fixed << std::setprecision(1) << confidence * 100.0 << "%)";
                        transcript_.push_back(oss.str());
                        lastPrediction_ = stablePrediction;
                        predictionCooldown_ = 0;
                        stabilityCounter_[stablePrediction] = 0; // Reset count for this specific word after adding
                    } else {
                        predictionCooldown_++; // Increment cooldown if no new stable word added
                    }
                    
                    // Prepare display text
                    std::ostringstream predOss;
                    predOss << "Pred: " << prediction << " (" << std::fixed << std::setprecision(2) << confidence << ")";
                    currentPredictionText = predOss.str();
                    std::ostringstream stabOss;
                    auto it = stabilityCounter_.find(prediction);
                    stabOss << "Stab: " << (it != stabilityCounter_.end() ? it->second : 0) << "/" << stabilityThreshold_;
                    stabilityText = stabOss.str();
                } else {
                    currentPredictionText = "Uncertain";
                    stabilityCounter_.clear(); // Clear stability if uncertain
                }
            }
        }
    }
    
    // --- Visualization ---
    cv::Scalar textColor = (currentPredictionText == "Uncertain" || currentPredictionText == "No motion detected") ?
        cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    
    cv::putText(visFrame, currentPredictionText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, textColor, 2);
    
    if (!stabilityText.empty()) {
        cv::putText(visFrame, stabilityText, cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
    }
    
    // Display motion score
    std::ostringstream motionOss;
    motionOss << "Motion: " << std::fixed << std::setprecision(0) << motionScore << "/" << motionThreshold_;
    cv::putText(visFrame, motionOss.str(), cv::Point(10, 180), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 0, 0), 2);
    
    // Display transcript (last 5 items, handle wrapping)
    cv::putText(visFrame, "Transcript:", cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    
    std::string transcriptLine;
    int startIdx = std::max(0, static_cast<int>(transcript_.size()) - 5);
    for (size_t i = startIdx; i < transcript_.size(); ++i) {
        transcriptLine += transcript_[i] + " ";
    }
    
    // Basic text wrapping
    int max_width = 60; // Approx characters
    int yPos = 120;
    std::string currentLine;
    std::istringstream iss(transcriptLine);
    std::string word;
    
    while (iss >> word) {
        if (currentLine.empty()) {
            currentLine = word;
        } else if (currentLine.length() + word.length() + 1 <= max_width) {
            currentLine += " " + word;
        } else {
            cv::putText(visFrame, currentLine, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            yPos += 30;
            currentLine = word;
        }
    }
    
    if (!currentLine.empty()) { // Print the last line
        cv::putText(visFrame, currentLine, cv::Point(10, yPos), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    }
    
    return visFrame;
}

void ASLInterpreter::saveTranscript(const std::string& filename) {
    std::ofstream ofs(filename);
    if (!ofs) {
        std::cerr << "Error: Could not open file for saving transcript: " << filename << std::endl;
        return;
    }
    
    for (size_t i = 0; i < transcript_.size(); ++i) {
        ofs << transcript_[i] << (i == transcript_.size() - 1 ? "" : " ");
    }
    
    ofs.close();
    std::cout << "Transcript saved to " << filename << std::endl;
    // Also save timestamped version
    std::string base = filename;
    std::string ext = "";
    size_t dotPos = filename.find_last_of('.');
    if (dotPos != std::string::npos) {
        base = filename.substr(0, dotPos);
        ext = filename.substr(dotPos); // Includes the dot
    }
    
    std::string timestampedFilename = base + "_" + getCurrentTimestamp() + ext;
    std::ofstream ofs_ts(timestampedFilename);
    if (!ofs_ts) {
        std::cerr << "Error: Could not open file for saving timestamped transcript: " << timestampedFilename << std::endl;
        return;
    }
    
    for (size_t i = 0; i < transcript_.size(); ++i) {
        ofs_ts << transcript_[i] << (i == transcript_.size() - 1 ? "" : " ");
    }
    
    ofs_ts.close();
    std::cout << "Transcript also saved to " << timestampedFilename << std::endl;
}

std::vector<std::string> ASLInterpreter::getTranscript() const {
    return transcript_;
}
