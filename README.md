Django app that uses Celery with Redis, processes videos with FFmpeg,
uses NLTK for text processing, and integrates with AWS services. 

## The views.py file in the videos app handles the end-to-end workflow of a video creation platform. 
# It lets authenticated users create and manage video projects by providing views for creating a project,writing or generating a script, processing that script into scenes, 
finding relevant media, generating voiceovers, and rendering the final video.
Each step is backed by asynchronous Celery tasks to handle time-consuming operations in the background. 
The views also manage project listing, deletion, and retrying failed renders, while using Django’s
messaging system to guide users through the multi-step creation process.

###This file defines a complete asynchronous video generation pipeline using Celery in a Django application.
It processes a user-submitted script by breaking it into scenes, generating media and voiceover audio for each scene using AI-powered services, and finally rendering a complete video.
The process is managed by the process_video_project task, which orchestrates four subtasks: script segmentation (process_script_task), 
media generation (find_media_task), voiceover creation (generate_voiceovers_task), 
and video rendering (render_video_task). The system includes robust error handling, file validation, 
retry logic with exponential backoff, and user email notifications on success or failure.

```
┌─────────────────────┐
│ process_video_project│
│   (Main Orchestrator)│
└────────────┬────────┘
             │
             ▼
 ┌────────────────────┐
 │ process_script_task│
 │   (Script → Scenes)│
 └────────────┬───────┘
              │
              ▼
     ┌────────────────┐
     │ find_media_task│
     │ (Generate Media│
     │   per Scene)   │
     └────────┬───────┘
              │
              ▼
 ┌────────────────────┐
 │generate_voiceovers_│
 │        task        │
 │ (Text → Voiceover) │
 └────────┬───────────┘
          │
          ▼
 ┌────────────────────┐
 │ render_video_task  │
 │ (Combine Media +   │
 │  Audio → Final MP4)│
 └────────┬───────────┘
          │
   ┌──────▼───────┐
   │ Email Notify │
   └──────────────┘
```

### This models.py file defines the data structure for an AI-powered video creation app, modeling how users create video projects starting from a script. It includes models for VideoProject (tracking project metadata and status), VideoScript (holding the script text and voice selection), Scene (breaking the script into ordered segments with extracted keywords), MediaAsset (storing generated images or videos for each scene), AudioAsset (holding the voiceover audio linked one-to-one with scenes), and RenderedVideo (representing the final combined video file tied to a project). Together, these models organize the full lifecycle of content generation, from text script to rendered video.

```
 User
   ▲
   │
VideoProject ─────────┐
     │                │
     │ 1-to-1         │ 1-to-1
     ▼                ▼
VideoScript      RenderedVideo
     │
     │ 1-to-many
     ▼
   Scene
   ▲   ▲
   │   │
   │   └───────────┐
   │               │
MediaAsset   AudioAsset
 (many)         (1)


```

### The forms.py file defines two Django forms used in the video creation workflow. VideoProjectForm is a ModelForm tied to the VideoProject model, allowing users to input a project title and description with a styled text area. ScriptForm is a regular form that collects the script text and a selected voice for narration. It dynamically populates voice choices using the VoiceGenerator service, which fetches available voice options (e.g., from a text-to-speech provider) if not explicitly provided. Together, these forms facilitate user input for generating AI-powered video projects.

```
| Feature            | Development Settings         | Production Settings                            |
| ------------------ | ---------------------------- | ---------------------------------------------- |
| `DEBUG`            | `True`                       | `False`                                        |
| Allowed Hosts      | Localhost IPs                | Env-based (`DJANGO_ALLOWED_HOSTS`)             |
| Debug Toolbar      | Enabled                      | Disabled                                       |
| Database           | PostgreSQL with dev defaults | PostgreSQL from environment vars               |
| Static Files       | Basic Django staticfiles     | WhiteNoise + compression + manifest            |
| Email Backend      | Console                      | Not defined (production email backend assumed) |
| Middleware Changes | None applied (commented out) | Adds `WhiteNoiseMiddleware`                    |
| Security Features  | None                         | Full HTTPS, Secure cookies, HSTS               |
| Logging            | None                         | Logs to console at WARNING level               |
| Sentry Monitoring  | Not included                 | Enabled                                        |



```


### This file sets up the Celery configuration for a Django project called videomaker. It initializes a Celery application, configures it to use Django’s settings (by reading environment variables prefixed with CELERY_), and enables automatic discovery of task modules (tasks.py) across all installed Django apps. The file also includes a sample task (debug_task) used for debugging purposes, which prints metadata about incoming task requests. This setup allows the project to run background tasks asynchronously using Celery.


### The services.py file is a Django application component that automates YouTube video creation through a pipeline of AI-driven services. It includes the ScriptGenerator class, which uses Google’s Gemini model to generate scripts with narration and scene descriptions, and the ScriptProcessor class, which segments scripts into scenes and extracts keywords using NLTK. The MediaFinder class generates images via Stability AI’s Stable Diffusion, while the VoiceGenerator class creates voiceovers with Amazon Polly, cleaning scripts to focus on narration. The VideoEditor class combines scenes into a final video using FFmpeg, with robust error handling and Django model integration for project tracking

