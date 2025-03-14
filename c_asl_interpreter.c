#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <unistd.h>
#include <sys/neutrino.h>  /* QNX specific */
#include <pthread.h>
#include <sched.h>
#include <opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <openvino/c/openvino.h>

/* Circular buffer to replace Python's deque */
typedef struct {
    void** data;
    int capacity;
    int count;
    int head;
    int tail;
    size_t element_size;
    void (*free_element)(void*);
} CircularBuffer;

/* Main structure to replace Python class */
typedef struct {
    /* OpenVINO model components */
    ov_core_t* core;
    ov_model_t* model;
    ov_compiled_model_t* compiled_model;
    ov_infer_request_t* infer_request;
    ov_tensor_t* input_tensor;
    ov_tensor_t* output_tensor;
    
    /* Model input parameters */
    int batch_size, channels, clip_length, height, width;
    
    /* Classes and buffers */
    char** classes;
    int class_count;
    CircularBuffer* frame_buffer;
    CircularBuffer* prediction_buffer;
    
    /* Thread synchronization */
    pthread_t inference_thread;
    pthread_mutex_t result_mutex;
    pthread_cond_t frame_ready_cond;
    bool inference_running;
    bool new_frames_available;
    
    /* Detection parameters and state */
    float confidence_threshold;
    int stability_threshold;
    int cooldown_threshold;
    int motion_threshold;
    
    /* Results tracking */
    char** transcript;
    int transcript_count;
    char last_prediction[64];
    int prediction_cooldown;
    int* stability_counter;
    
    /* Motion detection */
    IplImage* prev_gray;
    int no_motion_frames;
    
    /* Latest inference result */
    float* result_confidences;
    int* result_indices;
    int result_count;
} ASLInterpreter;

void* asl_inference_thread(void* arg) {
    ASLInterpreter* interpreter = (ASLInterpreter*)arg;
    
    /* Set thread to highest real-time priority */
    struct sched_param param;
    param.sched_priority = sched_get_priority_max(SCHED_RR);
    pthread_setschedparam(pthread_self(), SCHED_RR, &param);
    
    /* Set thread to run on a specific CPU core (for multicore systems) */
    int cpu_id = 0; /* Use first core for inference */
    ThreadCtl(_NTO_TCTL_RUNMASK, (void*)(1 << cpu_id));
    
    while (interpreter->inference_running) {
        pthread_mutex_lock(&interpreter->result_mutex);
        
        /* Wait for new frames or exit signal */
        while (!interpreter->new_frames_available && interpreter->inference_running) {
            pthread_cond_wait(&interpreter->frame_ready_cond, &interpreter->result_mutex);
        }
        
        if (!interpreter->inference_running) {
            pthread_mutex_unlock(&interpreter->result_mutex);
            break;
        }
        
        /* Process frames and run inference */
        if (interpreter->frame_buffer->count >= interpreter->clip_length) {
            process_frames_for_inference(interpreter);
            interpreter->new_frames_available = false;
        }
        
        pthread_mutex_unlock(&interpreter->result_mutex);
        
        /* Use nanosleep instead of usleep for more precise timing */
        struct timespec sleep_time = {0, 1000000}; /* 1ms */
        nanosleep(&sleep_time, NULL);
    }
    
    return NULL;
}

/* Pre-allocate buffers to avoid runtime allocations */
bool initialize_interpreter_buffers(ASLInterpreter* interpreter) {
    /* Pre-allocate input tensor memory */
    size_t input_size = interpreter->batch_size * interpreter->channels * 
                       interpreter->clip_length * interpreter->height * 
                       interpreter->width * sizeof(float);
    interpreter->input_data = (float*)memalign(64, input_size); /* Align for SIMD */
    
    /* Use memory pool for frame processing to avoid frequent allocations */
    interpreter->image_pool = create_image_pool(10, interpreter->width, interpreter->height);
    
    /* Use shared memory for faster inter-thread communication */
    interpreter->result_confidences = (float*)mmap(NULL, 
                                                MAX_CLASSES * sizeof(float),
                                                PROT_READ | PROT_WRITE,
                                                MAP_SHARED | MAP_ANON,
                                                NOFD, 0);
    interpreter->result_indices = (int*)mmap(NULL, 
                                           MAX_CLASSES * sizeof(int),
                                           PROT_READ | PROT_WRITE,
                                           MAP_SHARED | MAP_ANON,
                                           NOFD, 0);
    
    return (interpreter->input_data != NULL && interpreter->image_pool != NULL && 
            interpreter->result_confidences != NULL && interpreter->result_indices != NULL);
}

/* Process frame using SIMD optimizations when possible */
void process_frame_optimized(ASLInterpreter* interpreter, IplImage* frame) {
    pthread_mutex_lock(&interpreter->result_mutex);
    
    /* Motion detection using optimized functions */
    IplImage* gray = get_image_from_pool(interpreter->image_pool);
    cvCvtColor(frame, gray, CV_BGR2GRAY);
    
    /* Use SIMD-optimized Gaussian blur */
    IplImage* blurred = get_image_from_pool(interpreter->image_pool);
    gaussian_blur_simd(gray, blurred, 21, 21);
    
    int motion_score = 0;
    if (interpreter->prev_gray) {
        /* Use SIMD-optimized absolute difference */
        IplImage* frame_diff = get_image_from_pool(interpreter->image_pool);
        abs_diff_simd(interpreter->prev_gray, blurred, frame_diff);
        
        /* Threshold and calculate motion score */
        threshold_simd(frame_diff, frame_diff, 25, 255);
        motion_score = sum_pixels_simd(frame_diff) / 255;
        
        return_image_to_pool(interpreter->image_pool, frame_diff);
    }
    
    /* Update frame buffer */
    IplImage* frame_copy = cvCloneImage(frame);
    circular_buffer_push(interpreter->frame_buffer, frame_copy);
    
    /* Signal inference thread if we have enough frames */
    if (interpreter->frame_buffer->count >= interpreter->clip_length) {
        interpreter->new_frames_available = true;
        pthread_cond_signal(&interpreter->frame_ready_cond);
    }
    
    /* Clean up */
    if (interpreter->prev_gray) {
        return_image_to_pool(interpreter->image_pool, interpreter->prev_gray);
    }
    interpreter->prev_gray = blurred;
    return_image_to_pool(interpreter->image_pool, gray);
    
    pthread_mutex_unlock(&interpreter->result_mutex);
}

int main(int argc, char** argv) {
    /* Set process priority to high */
    setprio(0, 20);
    
    /* Lock memory to prevent paging */
    mlockall(MCL_CURRENT | MCL_FUTURE);
    
    /* Initialize message passing channel for UI updates */
    int chid = ChannelCreate(0);
    
    /* Initialize ASL interpreter */
    ASLInterpreter* interpreter = create_asl_interpreter(
        "asl-recognition-0004.xml",
        "asl-recognition-0004.bin",
        "MSASL_classes.json",
        "CPU",
        0.80f, 3, 45, 50
    );
    
    /* Start camera capture */
    CvCapture* capture = cvCaptureFromCAM(0);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
    cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);
    
    /* Start inference thread */
    start_inference_thread(interpreter);
    
    /* Main processing loop - run until terminated */
    struct sigevent event;
    struct itimerspec timer;
    timer.it_value.tv_sec = 0;
    timer.it_value.tv_nsec = 33333333; /* ~30 FPS */
    timer.it_interval = timer.it_value;
    
    int timer_id = TimerCreate(CLOCK_REALTIME, &event);
    TimerSettime(timer_id, 0, &timer, NULL);
    
    while (1) {
        /* Wait for timer or UI event */
        int rcvid = MsgReceive(chid, NULL, 0, NULL);
        
        /* Process frame */
        IplImage* frame = cvQueryFrame(capture);
        if (!frame) break;
        
        process_frame_optimized(interpreter, frame);
        
        /* Display results */
        IplImage* vis_frame = create_visualization(interpreter, frame);
        cvShowImage("ASL Interpreter", vis_frame);
        cvReleaseImage(&vis_frame);
        
        /* Handle key presses */
        int key = cvWaitKey(1);
        if (key == 'q') break;
    }
    
    /* Clean up */
    stop_inference_thread(interpreter);
    destroy_asl_interpreter(interpreter);
    cvReleaseCapture(&capture);
    
    return 0;
}
