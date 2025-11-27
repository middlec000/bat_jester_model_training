# Bat Jester
A program that counts the number of soccer ball juggles recorded in an audio file. The model developed in this repo is intended to be used in a phone application (code not stored in this repo).


## Environment Setup
- Clone the GitHub [repo](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).
    - Command line
    ```bash
    git clone https://github.com/middlec000/bat_jester_model_training.git
    ```
    - Or use the GitHub desktop app
- Use `uv` to manage Python and Python packages.
    - Install `uv` on your system - see [here](https://docs.astral.sh/uv/getting-started/installation/).
    - Ensure `uv` is installed:
    ```bash
    uv --version
    ```
    - Navigate to the project directory if you are not already in it
    ```bash
    cd .../bat_jester_model_training/
    ```
    - Use `uv` to install python and the packages for this project:
    ```bash
    uv sync
    ```
    - To add a new package:
    ```bash
    uv add <package_name>
    ```
- Download the data (see Contribute/Data Access) and store it in `/bat_jester_model_training/data/`
- Disregard the `flake.nix` and `flake.lock` files unless you are using `nix` for package management on your machine.

## Docs
- [Market Research](docs/market_research.md)
- [Development Plan](docs/development_plan.md)
- [Previous Approach](docs/previous_approach.md)

## Contribute
Would love to collaborate on this project!
- Data Access
    - [Email me](mailto:colindmiddleton@gmail.com)
- Communication Channel
    - [Google Chat Space](https://mail.google.com/chat/u/0/#chat/space/AAQAvpsTG20)
- Development git branching pattern
    - Branch from main with branch name describing branch purpose
    ```bash
    git switch main
    git branch clip_videos_on_voice_commands
    ```
    - When the work is ready, open a pull request in GitHub with target branch=`main`
