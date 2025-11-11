# /fred/oz413/ehunt/watermark_jobs/process_job.py
import os, sys, json, traceback
from core import generate_text

JOB_DIR = "/fred/oz413/ehunt/watermark_jobs"
INPUT_DIR = os.path.join(JOB_DIR, "input")
OUTPUT_DIR = os.path.join(JOB_DIR, "output")

def main(job_filename):
    job_path = os.path.join(INPUT_DIR, job_filename)
    output_path = os.path.join(OUTPUT_DIR, job_filename)

    try:
        with open(job_path) as f:
            job = json.load(f)

        result_text = generate_text(
            watermark=job["watermark"],
            prompt=job["prompt"],
            max_new_tokens=job["max_new_tokens"],
            key_file=job["key_file"],
            gen_model=job["model"],
            delta=job["delta"],
            entropy_threshold=job["entropy_threshold"],
            hashing_context=job["hashing_context"],
            user_id=job["user_id"],
            job_id=job_filename
        ) # gen job params

        with open(output_path, "w") as f:
            json.dump({"generated_text": result_text}, f)

    except Exception:
        err = traceback.format_exc()
        with open(output_path, "w") as f:
            json.dump({"error": err}, f)

if __name__ == "__main__":
    main(sys.argv[1])