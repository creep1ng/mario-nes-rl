from .cli import main

# The guard matters: worker processes started with "spawn" re-import this module
# and must not launch a second CLI.
if __name__ == '__main__':
    main()
