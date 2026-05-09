"""Streamlit GUI for the MOROCO topic-modeling project (assignment §2.3).

The app is a read-only frontend over the pipeline's saved artifacts in
`results/`. It never trains a model; the only "live" operation is
encoding a user-supplied text in the Try-it-live tab, which reuses the
already-trained models.
"""
