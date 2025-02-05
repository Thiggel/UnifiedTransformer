import itertools
import shutil

if __name__ == '__main__':
    dataset = ['mnist', 'fashion-mnist']
    image_embedding = ['conv', 'non-conv']
    embedding_dimension = [128, 256, 512]
    num_heads = [2, 4, 8]

    permutations = list(itertools.product(
        dataset,
        image_embedding,
        embedding_dimension,
        num_heads
    ))

    run_jobs_file_content = ''

    for permutation in permutations:
        dataset, image_embedding, embedding_dimension, num_heads = permutation
        print(dataset, image_embedding, embedding_dimension, num_heads)

        filename = f'jobs/jobdescription-{permutation}.sh' \
            .replace(' ', '_') \
            .replace("(", "") \
            .replace(")", "") \
            .replace(",", "") \
            .replace("'", "")

        # duplicate file 'jobs/jobdescription-sample.sh'
        shutil.copyfile('jobs/jobdescription-sample.sh', filename)

        with open(filename, 'r') as file:
            file_content = file.read()

        file_content = file_content.replace('%j',
                                            filename
                                            .replace('jobs/jobdescription-', '')
                                            .replace('.sh', '')
                                            )

        file_content += (
            f'\n'
            f'python3 main.py '
            f'--dataset {dataset} '
            f'--image-embedding {image_embedding} '
            f'--embedding-dimension {embedding_dimension} '
            f'--num-heads {num_heads}\n'
        )

        with open(filename, 'w') as file:
            file.write(file_content)

        run_jobs_file_content += f'sbatch {filename}\n'

    with open('run-jobs.sh', 'w') as file:
        file.write(run_jobs_file_content)
