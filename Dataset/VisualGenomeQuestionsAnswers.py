from json import load
from typing import Tuple, List, Optional, Callable, Any
from itertools import chain
from PIL import Image
from os.path import join, exists
from torch import Tensor
from time import time

from ImageTextDataset import ImageTextDataset


class VisualGenomeQuestionsAnswers(ImageTextDataset):

    def __init__(
            self,
            images_part1_dir: str,
            images_part2_dir: str,
            questions_answers_file: str,
            transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        print("Preprocessing Dataset")

        start = time()
        self.data = self.load_questions(questions_answers_file)
        self.questions = self.preprocess_text(self.data, text_key=1)
        print(f"Done! {time() - start}")
        
        # images are provided in two downloadable
        # packages, which is why we have to check
        # where an image is when loading it
        self.images_part1_dir = images_part1_dir
        self.images_part2_dir = images_part2_dir

        self.transform = transform

        # there is as many classes as words in the vocabulary
        # as an answer to a question can be any one word
        self.num_classes = self.vocab_size

    def preprocess_answer(self, answer: str) -> List:
        token_list = self.tokenizer(answer.replace('.', ''), return_tensors="pt").input_ids
        
        return token_list[0] if len(token_list) == 1 else token_list

    def preprocess_datapoint(self, datapoint) -> Tuple[int, str, Tensor]:
        answer = self.preprocess_answer(datapoint['answer'])
        # we filter out all the answers that contain
        # more than one word, as we don't train our model
        # on generating sentences but just classifying
        # single words
        if len(answer) == 3:
            return datapoint['image_id'], datapoint['question'], answer[1]

    def load_questions(self, question_answers_file: str) -> List[Tuple[int, str, Tensor]]:
        with open(question_answers_file, 'r') as file:
            json = load(file)

            return list(filter(None, [
                self.preprocess_datapoint(datapoint)
                for image in json
                for datapoint in image['qas']
            ]))

    def load_image(self, index: int) -> Image.Image:
        filename = f"{self.data[index][0]}.jpg"

        # as there are two folders with images provided
        # by the Visual Genome team, we want to save
        # the work of copying them into the same folder
        # and just check where our image exists
        dir1 = join(self.images_part1_dir, filename)
        dir2 = join(self.images_part2_dir, filename)

        directory = dir1 if exists(dir1) else dir2

        image = Image.open(directory).convert("RGB")

        return image

    def load_target(self, index: int) -> Tensor:
        return self.data[index][2]

    def __getitem__(self, index: int) -> Tuple[Tuple[Any, Tensor], Tensor]:
        # transform target word to numeric tensor using vocab
        answer = self.load_target(index)
        question = self.questions[index]

        # get image tensor
        image = self.load_image(index)

        if self.transform:
            image = self.transform(image)

        if isinstance(image, Tensor):
            image = image.float()
        return (image, question), answer

    def word_list(self) -> List[str]:
        return list(chain.from_iterable([datapoint[1:] for datapoint in self.data]))

    @property
    def sequence_length(self) -> int:
        return self.questions.shape[1]

    def __len__(self) -> int:
        return len(self.data)
