
# Motivation
#     I don't know, how to disable the recursive search of tensorboard.
#     Since to gets very slow, when there are to many folders and files
#     this script creates a symlink tree, which contains only the tfevents
#     files.
#     Additionally, the user can remove symlinks to remove them from the
#     tensorboard. The max_age option prevents that the symlinks are created
#     again, for old experiments.
#     The symlinks (vs copy) have the advantage, that tensorboard can still
#     track the appended data.


tree1days:
	find . -xtype l -delete  # Remove broken symlinks: https://unix.stackexchange.com/a/314975/283777
	python -m padertorch.contrib.cb.tensorboard_symlink_tree --prefix=.. ../*/*tfevents* --max_age=1days

tree7days:
	find . -xtype l -delete  # Remove broken symlinks: https://unix.stackexchange.com/a/314975/283777
	python -m padertorch.contrib.cb.tensorboard_symlink_tree --prefix=.. ../*/*tfevents* --max_age=7days

tree:
	find . -xtype l -delete  # Remove broken symlinks: https://unix.stackexchange.com/a/314975/283777
	python -m padertorch.contrib.cb.tensorboard_symlink_tree --prefix=.. ../*/*tfevents*

cleanup:
	# rm -f ???????????????/*
	rmdir --ignore-fail-on-non-empty *
