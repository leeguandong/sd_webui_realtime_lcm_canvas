import launch

if not launch.is_installed("diffusers"):
    try:
        launch.run_pip("install diffusers==0.24.0", "requirements for diffusers")
    except Exception:
        print("Can't install diffusers. Please follow the readme to install manually")
