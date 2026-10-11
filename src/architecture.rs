//! The two pinned, validated checkpoint architectures supported by this runtime.
use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelVariant {
    #[default]
    Clef,
    ClefFlash,
}
impl ModelVariant {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Clef => "clef",
            Self::ClefFlash => "clef-flash",
        }
    }
    pub const fn revision(self) -> &'static str {
        match self {
            Self::Clef => crate::checkpoint::REVISION,
            Self::ClefFlash => "8b2e5fd17c09fd49bc2880b805d5323ec7ff4ff3",
        }
    }
    pub const fn architecture(self) -> Architecture {
        match self {
            Self::Clef => Architecture {
                layers: 64,
                hidden: 5120,
                intermediate: 17408,
                attention_heads: 24,
                delta_heads: 48,
            },
            Self::ClefFlash => Architecture {
                layers: 32,
                hidden: 4096,
                intermediate: 12288,
                attention_heads: 16,
                delta_heads: 32,
            },
        }
    }
    pub(crate) fn reference(self, name: &str) -> &'static str {
        macro_rules! reference {
            ($file:literal) => {
                match self {
                    Self::Clef => include_str!(concat!("../reference/", $file)),
                    Self::ClefFlash => include_str!(concat!("../reference/clef-flash/", $file)),
                }
            };
        }
        match name {
            "config.json" => reference!("config.json"),
            "joint_head_config.json" => reference!("joint_head_config.json"),
            "processor_config.json" => reference!("processor_config.json"),
            "model.safetensors.index.json" => reference!("model.safetensors.index.json"),
            "backbone_metadata.json" => reference!("backbone_metadata.json"),
            "joint_head_metadata.json" => reference!("joint_head_metadata.json"),
            _ => unreachable!("unknown pinned metadata"),
        }
    }
}
#[derive(Debug, Clone, Copy)]
pub struct Architecture {
    pub layers: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub attention_heads: usize,
    pub delta_heads: usize,
}
impl Architecture {
    pub const fn embedding_bytes(self) -> u64 {
        (2 * 248320 * self.hidden * 2) as u64
    }
    pub const fn workspace_estimate(self, tokens: usize) -> u64 {
        tokens as u64 * self.intermediate as u64 * 2 * 12 + (2 << 30)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pinned_architecture_matches_tensor_metadata() {
        for model in [ModelVariant::Clef, ModelVariant::ClefFlash] {
            let arch = model.architecture();
            let config: serde_json::Value =
                serde_json::from_str(model.reference("config.json")).unwrap();
            let text = &config["text_config"];
            assert_eq!(text["hidden_size"], arch.hidden);
            assert_eq!(text["intermediate_size"], arch.intermediate);
            assert_eq!(text["num_hidden_layers"], arch.layers);
            assert_eq!(text["num_attention_heads"], arch.attention_heads);
            assert_eq!(text["linear_num_value_heads"], arch.delta_heads);
            let meta: serde_json::Value =
                serde_json::from_str(model.reference("backbone_metadata.json")).unwrap();
            assert_eq!(
                meta["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"]["shape"],
                serde_json::json!([4096 + arch.delta_heads * 128, arch.hidden])
            );
            assert_eq!(
                meta["model.visual.merger.linear_fc2.weight"]["shape"][0],
                arch.hidden
            );
            let head: serde_json::Value =
                serde_json::from_str(model.reference("joint_head_metadata.json")).unwrap();
            assert_eq!(
                head["memory_projection.weight"]["shape"],
                serde_json::json!([1024, arch.hidden])
            );
        }
    }
}
