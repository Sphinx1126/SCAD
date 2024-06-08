# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 05:49:11 2023

@author: 28257
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModel
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
import logger

class Adapter(nn.Module):
    def __init__(self,input_dim,bottleneck_dim,opt_dim=None):
        super(Adapter, self).__init__()
        self.input_dim=input_dim
        self.bottleneck_dim=bottleneck_dim
        self.opt_dim=self.input_dim if opt_dim is None else opt_dim
        self.down_project=nn.Linear(self.input_dim, self.bottleneck_dim)
        self.nonlinearity=nn.ReLU()
        self.up_project=nn.Linear(self.bottleneck_dim, self.opt_dim)
        
        torch.nn.init.normal_(self.down_project.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.up_project.weight, mean=0.0, std=0.01)
    
    def forward(self,x):
        opt=self.down_project(x)
        opt=self.nonlinearity(opt)
        opt=self.up_project(opt)
        opt+=x
        return opt

class AdaptedBERTLayer(nn.Module):
    def __init__(self, bert_layer,
                 input_dim,
                 #prompt_len,
                 bottleneck_dim,opt_dim=None):
        super(AdaptedBERTLayer, self).__init__()
        self.bert_layer=bert_layer
        for param in self.bert_layer.parameters():
            param.requires_grad=False
        for param in self.bert_layer.attention.output.LayerNorm.parameters():
            param.requires_grad=True
        for param in self.bert_layer.output.LayerNorm.parameters():
            param.requires_grad=True
        
        self.input_dim=input_dim
        #self.prompt_len=prompt_len
        #self.prompt=nn.Parameter(torch.zeros(self.prompt_len,self.input_dim), requires_grad=True)
        
        self.bottleneck_dim=bottleneck_dim
        self.opt_dim=self.input_dim if opt_dim is None else opt_dim
        self.adapter1=Adapter(self.input_dim,self.bottleneck_dim)
        self.adapter2=Adapter(self.input_dim,self.bottleneck_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        #bs=hidden_states.size(0)
        #prompt=self.prompt.unsqueeze(0).expand(bs,self.prompt_len,self.input_dim)
        #hidden_states=torch.cat([hidden_states,prompt],dim=1)
        
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.bert_layer.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        
        attention_output = self_attention_outputs[0]
        attention_output=self.adapter1(attention_output)

        # if decoder, the last output is tuple of self-attn cache
        if self.bert_layer.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.bert_layer.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self.bert_layer, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.bert_layer.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.bert_layer.feed_forward_chunk, self.bert_layer.chunk_size_feed_forward, self.bert_layer.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.bert_layer.is_decoder:
            outputs = outputs + (present_key_value,)

        return (outputs[0],)

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.bert_layer.intermediate(attention_output)
        layer_output = self.bert_layer.output(intermediate_output, attention_output)
        
        layer_output=self.adapter2(layer_output)
        
        return layer_output
     
class AdaptedBERTEncoder(nn.Module):
    def __init__(self, bert_encoder,
                 input_dim,
                 #prompt_len,
                 bottleneck_dim,opt_dim=None):
        super(AdaptedBERTEncoder, self).__init__()
        
        self.input_dim=input_dim
        #self.prompt_len=prompt_len
        self.bottleneck_dim=bottleneck_dim
        self.opt_dim=self.input_dim if opt_dim is None else opt_dim
        
        self.config = bert_encoder.config
        self.layer = nn.ModuleList([AdaptedBERTLayer(bert_encoder.layer[i],
                                                     self.input_dim,
                                                     self.bottleneck_dim,
                                                     opt_dim=self.opt_dim if i==len(bert_encoder.layer)-1 else self.input_dim) 
                                    for i in range(len(bert_encoder.layer))])
        self.gradient_checkpointing = bert_encoder.gradient_checkpointing

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        



class AdaptedBERT(nn.Module):
    def __init__(self, pretrained_bert,
                 #prompt_len,
                 bottleneck_dim,opt_dim):
        super(AdaptedBERT, self).__init__()
        for param in pretrained_bert.parameters():
            param.requires_grad=False
        self.config = pretrained_bert.config
        self.embeddings = pretrained_bert.embeddings
        self.pooler = pretrained_bert.pooler
        self.get_extended_attention_mask=pretrained_bert.get_extended_attention_mask
        self.invert_attention_mask=pretrained_bert.invert_attention_mask
        self.get_head_mask=pretrained_bert.get_head_mask
        
        self.hidden_size=pretrained_bert.config.hidden_size
        #self.prompt_len=prompt_len
        self.bottleneck_dim=bottleneck_dim
        self.opt_dim=opt_dim
        self.encoder = AdaptedBERTEncoder(pretrained_bert.encoder,
                                          self.hidden_size,
                                          self.bottleneck_dim,
                                          self.opt_dim)
        
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return pooled_output




if __name__=='__main__':
    pretrained_bert=AutoModel.from_pretrained("bert-base-uncased")
    perta=AdaptedBERT(pretrained_bert,2,8,64)
    
    len1=128
    len2=255
    text1=[514]*len1+[103]*(512-len1)
    text2=[114]*len2+[103]*(512-len2)
    mask1=[1]*(len1+2)+[0]*(512-len1)
    mask2=[1]*(len2+2)+[0]*(512-len2)
    texts=torch.tensor([text1,text2])
    masks=torch.tensor([mask1,mask2])
    
    perta(input_ids=texts,attention_mask=masks)