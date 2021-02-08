import os
import torch
from typing import Optional, Union, List
from torch.nn import functional as F
from transformers.modeling_rag import RagSequenceForGeneration, RagRetriever, RagTokenForGeneration
from transformers.tokenization_rag import RagTokenizer, RagConfig
from transformers.generation_beam_search import BeamSearchScorer, BeamScorer
from transformers.generation_logits_process import LogitsProcessorList
from retriever import RagDprRetriever
from time import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RagSequenceGenerator(RagSequenceForGeneration):

    @classmethod
    def from_pretrained(cls, model_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        print(
            '############################### RagSequenceGenerator.from_pretrained begin #######################################')
        print('generator.py:12:from_pretrained begin')
        t = time()
        if 'retriever' not in kwargs or not kwargs['retriever']:
            # retriever = DprRagRetriever.from_pretrained(model_path)
            retriever = RagDprRetriever(model_path)
            kwargs.update({'retriever': retriever})
            print(f'{time() - t} seconds to load retriever')
            t = time()
        print('...')
        model = super(RagSequenceGenerator, cls).from_pretrained(model_path, *model_args, **kwargs)
        print('moving model to device ', device)
        model.to(device)
        model.tokenizer = RagTokenizer.from_pretrained(model_path)
        print(f'generator.py:23:from_pretrained finihed: {time() - t} seconds to load encoder and generator models')
        print(
            '############################# RagSequenceGenerator.from_pretrained finished #####################################')
        return model

    @torch.no_grad()
    def predict(self, questions: List[str]):
        print('########################### generator.predict begin ##########################################')
        print('generator.py:30:predict begin')
        t = time()
        input_dict = self.tokenizer.prepare_seq2seq_batch(questions, return_tensors="pt")
        input_dict.to(device)
        input_ids = input_dict["input_ids"]

        n_docs = self.config.n_docs
        do_deduplication = self.config.do_deduplication
        num_beams = self.config.num_beams
        # n_docs = 20
        # num_doc_return_sequences = 6
        # num_beams = 10
        print(f'n_docs: {n_docs}')
        print(f'do_deduplication: {do_deduplication}')
        print(f'num_beams: {num_beams}')

        question_encoder_last_hidden_state = self.question_encoder(input_ids)[0]
        retriever_outputs = self.retriever(
            input_ids,
            question_encoder_last_hidden_state.cpu().detach().to(torch.float32).numpy(),
            prefix=self.generator.config.prefix,
            n_docs=n_docs,
            return_tensors="pt",
        )
        context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_ids = (
            retriever_outputs["context_input_ids"],
            retriever_outputs["context_attention_mask"],
            retriever_outputs["retrieved_doc_embeds"],
            retriever_outputs["doc_ids"],
        )
        print('context_input_ids', self.tokenizer.batch_decode(context_input_ids, skip_special_tokens=True))
        print(f'context_input_ids.shape: {context_input_ids.shape}')
        print(f'context_attention_mask.shape: {context_attention_mask.shape}')
        print(f'retrieved_doc_embeds.shape: {retrieved_doc_embeds.shape}')
        # set to correct device
        context_input_ids = context_input_ids.to(input_ids)
        context_attention_mask = context_attention_mask.to(input_ids)
        retrieved_doc_embeds = retrieved_doc_embeds.to(question_encoder_last_hidden_state)
        # compute doc_scores
        doc_scores = torch.bmm(
            question_encoder_last_hidden_state.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
        ).squeeze(1)

        model_kwargs = {
            'num_beams': num_beams,
            'num_return_sequences': num_beams,
            'attention_mask': None
        }
        print(f'model_kwargs: {model_kwargs}')
        results = []

        for index in range(len(questions)):
            # first, generate beams from documents:
            generator_input_ids = context_input_ids[index * n_docs: (index + 1) * n_docs]  # (n_docs, max_len)

            output_sequences = self.generator.generate(
                generator_input_ids,
                **model_kwargs,
            )  # n_docs * n_beam, tgt_len

            if do_deduplication:
                # do_deduplication, max_output_len
                output_sequences = torch.stack(list({str(k.tolist()): k for k in output_sequences}.values()))

            # then, run model forwards to get nll scores:
            new_input_ids = input_ids[index: index + 1].repeat(len(output_sequences), 1)
            new_context_input_ids = None
            new_context_attention_mask = None
            new_doc_scores = None
            outputs = self.forward(new_input_ids, labels=output_sequences, exclude_bos_score=True,
                                   context_input_ids=new_context_input_ids,
                                   context_attention_mask=new_context_attention_mask,
                                   doc_scores=new_doc_scores)
            loss_scores = outputs['loss'].cpu().detach().to(torch.float32).numpy()

            print('index:', index)
            print('input_ids:', self.tokenizer.question_encoder.batch_decode(input_ids, skip_special_tokens=True))
            print('context_input_ids', self.tokenizer.batch_decode(context_input_ids, skip_special_tokens=True))
            print('generator_input_ids', self.tokenizer.batch_decode(generator_input_ids, skip_special_tokens=True))
            print('output_sequences: ', self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True))
            print('new_input_ids', self.tokenizer.batch_decode(new_input_ids, skip_special_tokens=True))
            print('top output sequence', self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True))

            encoded_answers = self._cat_and_pad([output_sequences], pad_token_id=self.config.generator.pad_token_id)
            decoded_answers = self.tokenizer.batch_decode(encoded_answers, skip_special_tokens=True)
            answers = []
            for i, a in enumerate(decoded_answers):
                answers.append({
                    'text': a,
                    'loss_score': loss_scores[i]
                })
            answers.sort(key=lambda x: x['loss_score'])
            results.append({
                'question': questions[index],
                'answers': answers
            })
        print('generator.py:30:predict end' + f'{time() - t} seconds to predict')
        print('########################### generator.predict finished ##########################################')
        return results


class RagTokenGenerator(RagTokenForGeneration):

    @classmethod
    def from_pretrained(cls, model_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        print('########################## RagTokenGenerator.from_pretrained begin ##################################')
        t = time()
        if 'retriever' not in kwargs or not kwargs['retriever']:
            retriever = RagDprRetriever(model_path)
            kwargs.update({'retriever': retriever})
            print(f'{time() - t} seconds to load retriever')
            t = time()
        print('...')
        model = super(RagTokenGenerator, cls).from_pretrained(model_path, *model_args, **kwargs)
        print('moving model to device ', device)
        model.to(device)
        model.tokenizer = RagTokenizer.from_pretrained(model_path)
        print(f'from_pretrained finihed: {time() - t} seconds to load encoder and generator models')
        print('######################### RagTokenGenerator.from_pretrained finished ################################')
        return model

    def beam_search(
            self,
            input_ids: torch.LongTensor,
            beam_scorer: BeamScorer,
            logits_processor: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            **model_kwargs
    ):
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        max_length = max_length if max_length is not None else self.config.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, cur_len = input_ids.shape

        assert (
                num_beams * batch_size == batch_beam_size
        ), "Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))
        output_scores = []

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(**model_inputs, return_dict=True)
            next_token_logits = outputs.logits[:, -1, :]

            # adjust tokens for Bart, *e.g.*
            next_token_logits = self.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

            next_token_scores = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            cur_len = cur_len + 1

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)

            if beam_scorer.is_done:
                break

        decoded = beam_scorer.finalize(
            input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id, eos_token_id=eos_token_id
        )
        # beam_scores.cpu().detach().to(torch.float32).numpy()
        return decoded

    @torch.no_grad()
    def predict(self, questions: List[str]):
        # set default parameters
        n_docs = self.config.n_docs
        num_beams = self.config.num_beams
        max_length = self.config.max_length
        num_return_sequences = (
            self.config.num_return_sequences
        )
        num_return_sequences = 1
        print('n_docs: ', n_docs)
        print('num_beams: ', num_beams)
        print('max_length: ', max_length)
        print('num_return_sequences: ', num_return_sequences)
        bos_token_id = self.config.generator.bos_token_id
        eos_token_id = self.config.generator.eos_token_id
        pad_token_id = self.config.generator.pad_token_id
        decoder_start_token_id = (
            self.config.generator.decoder_start_token_id
        )

        # input ids
        input_dict = self.tokenizer.prepare_seq2seq_batch(questions, return_tensors="pt")
        input_dict.to(device)
        input_ids = input_dict["input_ids"]

        # retrieve docs
        question_hidden_states = self.question_encoder(input_ids, attention_mask=None)[0]
        out = self.retriever(
            input_ids,
            question_hidden_states.cpu().detach().to(torch.float32).numpy(),
            prefix=self.generator.config.prefix,
            n_docs=n_docs,
            return_tensors="pt",
        )
        context_input_ids, context_attention_mask, retrieved_doc_embeds = (
            out["context_input_ids"],
            out["context_attention_mask"],
            out["retrieved_doc_embeds"],
        )

        # set to correct device
        retrieved_doc_embeds = retrieved_doc_embeds.to(question_hidden_states)
        context_input_ids = context_input_ids.to(input_ids)
        context_attention_mask = context_attention_mask.to(input_ids)

        # compute doc_scores
        doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(
            1
        )

        assert (
                       context_input_ids.shape[0] % n_docs
               ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # batch_size
        batch_size = context_input_ids.shape[0] // n_docs

        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask, return_dict=True)

        input_ids = torch.full(
            (batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            # split into `batch_size`, `num_beams`, `num_docs`
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            # repeat same last hidden states over `num_beams` dimension
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            # merge `batch_size`, `num_beams`, `num_docs` dims again
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])

        # correctly extend last_hidden_state and attention mask
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(last_hidden_state, num_beams=num_beams)

        doc_scores = doc_scores.repeat_interleave(num_beams, dim=0)

        # define start_len & additional parameters
        model_kwargs = {}
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs

        repetition_penalty = None
        no_repeat_ngram_size = None
        bad_words_ids = None
        min_length = None
        pre_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            eos_token_id=eos_token_id,
        )

        if num_beams == 1:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            outputs = self.greedy_search(
                input_ids,
                pre_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
            decoded_answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return decoded_answers
        elif num_beams > 1:
            length_penalty = self.config.length_penalty
            early_stopping = self.config.early_stopping
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                max_length=max_length,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            outputs = self.beam_search(
                input_ids,
                beam_scorer,
                pre_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
            decoded_answers = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            results = []
            for i in range(len(questions)):
                answers = []
                for j in range(num_return_sequences):
                    answers.append({
                        'text': decoded_answers[i*num_return_sequences + j]
                    })
                # answers.sort(key=lambda x: x['loss_score'], reverse=True)
                results.append({
                    'question': questions[i],
                    'answers': answers
                })
            return results
        else:
            raise ValueError(f"`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {num_beams}")
