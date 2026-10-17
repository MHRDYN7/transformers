# Investigation: Audio Models Using Mimi from Moshi

## Executive Summary

This document details the investigation of audio models in the Hugging Face Transformers library that use or inherit from **Mimi**, the neural audio codec that is part of the **Moshi** speech-text foundation model.

## Background

### What is Mimi?

**Mimi** is a neural audio codec model designed for efficient speech representation and compression. Key features:
- Operates at 1.1 kbps with a 12 Hz frame rate
- Uses convolutional encoder-decoder architecture with residual vector quantization (16 codebooks)
- Outputs dual token streams (semantic and acoustic) for linguistic richness and high-fidelity reconstruction
- Designed for causal streaming with low latency
- Released by Kyutai on 2024-09-17, added to Transformers on 2024-09-18

### What is Moshi?

**Moshi** is a speech-text foundation model for real-time dialogue that:
- Uses Mimi as its audio encoder/decoder component
- Provides full-duplex spoken dialogue capabilities
- Models both user and assistant speech in parallel streams
- Achieves ~160ms theoretical latency (200ms in practice)
- Includes "Inner Monologue" method for text-aligned speech generation

## Models Using Mimi Components

### 1. **Moshi** (`src/transformers/models/moshi/`)

**Relationship**: Uses Mimi as the default audio encoder

- **Configuration**: `MoshiConfig` uses Mimi as the default audio encoder (line 307 in `configuration_moshi.py`)
  ```python
  audio_encoder_model_type = audio_encoder_config.pop("model_type", "mimi")
  ```
- **Code Copying**: Moshi copies `MimiSdpaAttention` with replacements (line 636 in `modeling_moshi.py`)
  ```python
  # Copied from transformers.models.mimi.modeling_mimi.MimiSdpaAttention with Mimi->Moshi
  ```
- **Purpose**: Moshi uses Mimi to encode/decode audio for its full-duplex speech dialogue system

### 2. **KyutaiSpeechToText** (`src/transformers/models/kyutai_speech_to_text/`)

**Relationship**: Inherits from Moshi and uses Mimi components

- **Imports**:
  - `MimiConv1dPaddingCache` from `mimi.modeling_mimi` (line 30)
  - `MoshiModel`, `MoshiPreTrainedModel` from `moshi.modeling_moshi` (line 31)

- **Class Inheritance**:
  - `KyutaiSpeechToTextPreTrainedModel(MoshiPreTrainedModel)` - Inherits from Moshi's base class
  - `KyutaiSpeechToTextConv1dPaddingCache(MimiConv1dPaddingCache)` - Inherits Mimi's padding cache
  - `KyutaiSpeechToTextModel(MoshiModel)` - Inherits from Moshi's main model
  - Uses `LlamaForCausalLM` for the generation component

- **Purpose**: Speech-to-text model that leverages Moshi architecture and Mimi's audio processing

### 3. **Qwen3OmniMoe** (`src/transformers/models/qwen3_omni_moe/`)

**Relationship**: Uses Mimi's LayerScale component

- **Import**: `MimiLayerScale` from `mimi.modeling_mimi` (line 51)

- **Class Inheritance**:
  - `Qwen3OmniMoeCode2WavLayerScale(MimiLayerScale)` (line 2303)

- **Purpose**: Multimodal model that uses Mimi's layer scaling technique in its code-to-wav conversion component

### 4. **VibeVoiceAcousticTokenizer** (`src/transformers/models/vibevoice_acoustic_tokenizer/`)

**Relationship**: Uses Mimi's padding cache

- **Import**: `MimiConv1dPaddingCache` from `mimi.modeling_mimi` (line 26)

- **Class Inheritance**:
  - `VibeVoiceAcousticTokenizerConv1dPaddingCache(MimiConv1dPaddingCache)` (line 93)

- **Purpose**: Acoustic tokenizer that uses Mimi's convolutional padding cache for audio processing

### 5. **CSM** (`src/transformers/models/csm/`)

**Relationship**: Uses Mimi as the codec model

- **Import**: `MimiModel` in conversion script (line 30 in `convert_csm.py`)

- **Usage**: 
  ```python
  codec_model = MimiModel.from_pretrained(codec_model_path_or_repo)
  ```

- **Purpose**: CSM model uses Mimi as its codec for audio encoding/decoding during model conversion

## Component Inheritance Map

```
Mimi (Audio Codec)
├── Moshi (Uses Mimi as audio encoder)
│   └── KyutaiSpeechToText (Inherits from Moshi)
├── Qwen3OmniMoe (Uses MimiLayerScale)
├── VibeVoiceAcousticTokenizer (Uses MimiConv1dPaddingCache)
└── CSM (Uses MimiModel for codec)
```

## Mimi Components Used by Other Models

### 1. **MimiConv1dPaddingCache**
- Used by: **KyutaiSpeechToText**, **VibeVoiceAcousticTokenizer**
- Purpose: Manages padding cache for causal convolution operations in streaming scenarios

### 2. **MimiLayerScale**
- Used by: **Qwen3OmniMoe**
- Purpose: Provides layer scaling functionality for stable training

### 3. **MimiModel**
- Used by: **Moshi** (as audio encoder), **CSM** (as codec)
- Purpose: Full audio encoding/decoding model

### 4. **MimiSdpaAttention**
- Copied by: **Moshi** (as `MoshiSdpaAttention`)
- Purpose: Scaled dot-product attention implementation

## Audio Model Categories

Based on this investigation, audio models can be categorized as:

1. **Direct Mimi Users**: 
   - Moshi (uses Mimi as audio encoder)
   - CSM (uses Mimi as codec)

2. **Moshi Inheritors**:
   - KyutaiSpeechToText (inherits from Moshi architecture)

3. **Mimi Component Users**:
   - Qwen3OmniMoe (uses MimiLayerScale)
   - VibeVoiceAcousticTokenizer (uses MimiConv1dPaddingCache)
   - KyutaiSpeechToText (also uses MimiConv1dPaddingCache)

4. **Other Audio Models** (not using Mimi/Moshi):
   - audio_spectrogram_transformer
   - audioflamingo3
   - clap
   - encodec
   - fastspeech2_conformer
   - granite_speech
   - hubert
   - pe_audio
   - pe_audio_video
   - qwen2_audio
   - seamless_m4t
   - seamless_m4t_v2
   - speech_encoder_decoder
   - speech_to_text
   - speecht5
   - unispeech
   - unispeech_sat
   - wav2vec2
   - wav2vec2_bert
   - wav2vec2_conformer
   - wav2vec2_phoneme
   - wav2vec2_with_lm

## Key Findings

1. **Mimi is the audio codec for Moshi**: Moshi's default audio encoder is Mimi, making it the foundation of Moshi's audio processing capabilities.

2. **KyutaiSpeechToText has the deepest integration**: It inherits from both Moshi and uses Mimi components directly, making it the most tightly coupled model.

3. **Modular design**: Specific Mimi components (LayerScale, Conv1dPaddingCache) are reused across different models, showing good architectural modularity.

4. **Limited adoption**: Out of 25+ audio models in transformers, only 4-5 models use Mimi or Moshi components, indicating that Mimi/Moshi are relatively new additions or specialized for specific use cases.

5. **Kyutai ecosystem**: Both Mimi and Moshi are from Kyutai, and KyutaiSpeechToText also appears to be part of the same ecosystem, suggesting a cohesive family of models.

## File Locations

### Mimi
- Configuration: `src/transformers/models/mimi/configuration_mimi.py`
- Model: `src/transformers/models/mimi/modeling_mimi.py`
- Documentation: `docs/source/en/model_doc/mimi.md`

### Moshi  
- Configuration: `src/transformers/models/moshi/configuration_moshi.py`
- Model: `src/transformers/models/moshi/modeling_moshi.py`
- Documentation: `docs/source/en/model_doc/moshi.md`

### Models Using Mimi/Moshi
- KyutaiSpeechToText: `src/transformers/models/kyutai_speech_to_text/modular_kyutai_speech_to_text.py`
- Qwen3OmniMoe: `src/transformers/models/qwen3_omni_moe/modular_qwen3_omni_moe.py`
- VibeVoiceAcousticTokenizer: `src/transformers/models/vibevoice_acoustic_tokenizer/modular_vibevoice_acoustic_tokenizer.py`
- CSM: `src/transformers/models/csm/convert_csm.py`

## Conclusion

Mimi serves as a foundational audio codec in the transformers library, primarily supporting the Moshi speech-text model. While it's used by a small number of models, those that do use it demonstrate significant architectural integration, particularly KyutaiSpeechToText which inherits from Moshi. The modular design of Mimi allows for selective reuse of its components (like LayerScale and Conv1dPaddingCache) across different audio models.
