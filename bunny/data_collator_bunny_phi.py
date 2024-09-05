from dataclasses import dataclass
from typing import Dict, List, Any
from trl.trainer.utils import DPODataCollatorWithPadding
from PIL import Image

from bunny.bunny_utils.util.mm_utils import tokenizer_image_token


@dataclass
class mDPODataCollatorBunny(DPODataCollatorWithPadding):
    def tokenize_batch_element(
        self,
        prompt: str,
        chosen: str,
        rejected: str,
        img_path: str,
    ) -> Dict:
        batch = {}

        chosen_tokens = self.tokenizer(chosen, add_special_tokens=False)
        rejected_tokens = self.tokenizer(rejected, add_special_tokens=False)
        prompt_tokens = {}
        prompt_tokens["input_ids"] = tokenizer_image_token(prompt, self.tokenizer)
        prompt_tokens["attention_mask"] = [1 for _ in range(len(prompt_tokens["input_ids"]))]

        eos_token_id = self.tokenizer.eos_token_id
        # Get indices in list prompt_tokens["input_ids"] that equals the EOS token (often 0)
        eos_indices_prompt = [i for i, x in enumerate(prompt_tokens["input_ids"]) if x == eos_token_id]
        # attention mask these indices to eos_token_id
        new_attention_mask = [
            0 if i in eos_indices_prompt else p for i, p in enumerate(prompt_tokens["attention_mask"])
        ]
        prompt_tokens["attention_mask"] = new_attention_mask

        # do the same for chosen and rejected
        eos_indices_chosen = [i for i, x in enumerate(chosen_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_c = [
            0 if i in eos_indices_chosen else p for i, p in enumerate(chosen_tokens["attention_mask"])
        ]
        chosen_tokens["attention_mask"] = new_attention_mask_c

        eos_indices_rejected = [i for i, x in enumerate(rejected_tokens["input_ids"]) if x == eos_token_id]
        new_attention_mask_r = [
            0 if i in eos_indices_rejected else p for i, p in enumerate(rejected_tokens["attention_mask"])
        ]
        rejected_tokens["attention_mask"] = new_attention_mask_r

        # add EOS token to end of prompt
        chosen_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        chosen_tokens["attention_mask"].append(1)

        rejected_tokens["input_ids"].append(self.tokenizer.eos_token_id)
        rejected_tokens["attention_mask"].append(1)

        longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

        # if combined sequence is too long, truncate the prompt
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            if self.truncation_mode == "keep_start":
                prompt_tokens = {k: v[: self.max_prompt_length] for k, v in prompt_tokens.items()}
            elif self.truncation_mode == "keep_end":
                prompt_tokens = {k: v[-self.max_prompt_length :] for k, v in prompt_tokens.items()}
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

        # if that's still too long, truncate the response
        if len(prompt_tokens["input_ids"]) + longer_response_length > self.max_length:
            chosen_tokens = {k: v[: self.max_length - self.max_prompt_length] for k, v in chosen_tokens.items()}
            rejected_tokens = {
                k: v[: self.max_length - self.max_prompt_length] for k, v in rejected_tokens.items()
            }

        # Create labels
        chosen_sequence_tokens = {k: prompt_tokens[k] + chosen_tokens[k] for k in chosen_tokens}
        rejected_sequence_tokens = {k: prompt_tokens[k] + rejected_tokens[k] for k in rejected_tokens}
        chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
        chosen_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )
        rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
        rejected_sequence_tokens["labels"][: len(prompt_tokens["input_ids"])] = [self.label_pad_token_id] * len(
            prompt_tokens["input_ids"]
        )

        for k, toks in {
            "chosen": chosen_sequence_tokens,
            "rejected": rejected_sequence_tokens,
            "prompt": prompt_tokens,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens

        image = Image.open(img_path)
        image_tensor = self.model.process_images([image], self.model.config).to(dtype=self.model.dtype)
        batch["image"] = image_tensor

        return batch
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            tokenized_batch = []

            for feature in features:
                prompt = feature["prompt"]
                chosen = feature["chosen"]
                rejected = feature["rejected"]
                img_path = feature["img_path"]

                batch_element = self.tokenize_batch_element(prompt, chosen, rejected, img_path)
                tokenized_batch.append(batch_element)

            collated_batch = self.collate(tokenized_batch)
            return collated_batch
