#! /bin/sh
python random_indexing.py -c -co cleaned_example.txt
diff --strip-trailing-cr cleaned_example.txt correct_cleaned_example.txt