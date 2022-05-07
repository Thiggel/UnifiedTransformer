import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from VisualGenomeQuestionsAnswers import VisualGenomeQuestionsAnswers
from ImageTextDataModule import ImageTextDataModule


class VisualGenomeDataModule(ImageTextDataModule):
    def __init__(self, batch_size: int = 32) -> None:
        super().__init__()

        self.batch_size = batch_size

        self.images_part1_dir = '../../../Fritz/Multi-Modal-Transformer/VG_100K'
        self.images_part1_url = 'https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip'

        self.images_part2_dir = './../../../Fritz/Multi-Modal-Transformer/VG_100K_2'
        self.images_part2_url = 'https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip'

        self.questions_answers_file = './../../../Fritz/Multi-Modal-Transformer/question_answers.json'
        self.questions_answers_url = 'https://visualgenome.org/static/data/dataset/question_answers.json.zip'

        # download dataset if not yet downloaded
        self.load_dataset()

        visual_genome_full = VisualGenomeQuestionsAnswers(
            images_part1_dir=self.images_part1_dir,
            images_part2_dir=self.images_part2_dir,
            questions_answers_file=self.questions_answers_file,
            transform=transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.PILToTensor()
            ])
        )

        self.vocab_size = visual_genome_full.vocab_size
        self.sequence_length = visual_genome_full.sequence_length
        self.num_classes = visual_genome_full.num_classes

        # split into train/test/val with 70/20/10 ratio
        self.visual_genome_train, \
        self.visual_genome_test, \
        self.visual_genome_val = self.split_dataset(visual_genome_full)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.visual_genome_train, batch_size=self.batch_size, shuffle=True, num_workers=12)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.visual_genome_val, batch_size=self.batch_size, num_workers=12)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.visual_genome_test, batch_size=self.batch_size, num_workers=12)

    def load_dataset(self) -> None:
        self.download_if_not_exists(self.images_part1_dir, self.images_part1_url)
        self.download_if_not_exists(self.images_part2_dir, self.images_part2_url)
        self.download_if_not_exists(self.questions_answers_file, self.questions_answers_url)


VisualGenomeDataModule()