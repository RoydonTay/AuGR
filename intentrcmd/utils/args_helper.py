import argparse
import os

from intentrcmd.datasets.base_data_processor import BaseDataProcessor
from intentrcmd.datasets.smartkb_data_processor import SmartKBDataProcessor

from intentrcmd.modules.ip_fusion_model import IPFusionModel
from intentrcmd.modules.ip_dcn_model import IPDCNv2Model
from intentrcmd.modules.ip_fusion_dcn_model import IPFusionDCNv2Model
from intentrcmd.modules.ip_fusion_debias_model import IPFusionDebiasModel
from intentrcmd.modules.ip_dcn_debias_model import IPDCNv2DebiasModel
from intentrcmd.modules.ip_bifusion_debias_model import IPBiFusionDebiasModel
from intentrcmd.modules.ip_fusion_3tower_model import IPFusion3TowerModel
from intentrcmd.modules.ip_fusion_debias_3tower_model import IPFusionDebias3TowerModel
from intentrcmd.modules.ip_fusion_rankmixer_model import IPFusionRankMixerModel
from intentrcmd.modules.ip_fusion_group_rankmixer_model import IPFusionGroupRankMixerModel
from intentrcmd.modules.ip_fusion_debias_user_rankmixer_model import IPFusionDebiasUserRankMixerModel
from intentrcmd.modules.ip_fusion_debias_group_rankmixer_model import IPFusionDebiasGroupRankMixerModel
from intentrcmd.modules.ip_fusion_group_tokenmixer_model import IPFusionGroupTokenMixerModel
from intentrcmd.modules.ip_fusion_group_rankmixer_debias_model import IPFusionGroupRankMixerDebiasModel


MODEL_CLS_DICT = {
    'IPFusionModel': IPFusionModel,
    'IPDCNv2Model': IPDCNv2Model,
    'IPFusionDCNv2Model': IPFusionDCNv2Model,
    'IPFusionDebiasModel': IPFusionDebiasModel,
    'IPDCNv2DebiasModel': IPDCNv2DebiasModel,
    'IPBiFusionDebiasModel': IPBiFusionDebiasModel,
    'IPFusion3TowerModel': IPFusion3TowerModel,
    'IPFusionDebias3TowerModel': IPFusionDebias3TowerModel,
    'IPFusionRankMixerModel': IPFusionRankMixerModel,
    'IPFusionGroupRankMixerModel': IPFusionGroupRankMixerModel,
    'IPFusionDebiasUserRankMixerModel': IPFusionDebiasUserRankMixerModel,
    'IPFusionDebiasGroupRankMixerModel': IPFusionDebiasGroupRankMixerModel,
    'IPFusionGroupTokenMixerModel': IPFusionGroupTokenMixerModel,
    'IPFusionGroupRankMixerDebiasModel': IPFusionGroupRankMixerDebiasModel,
}

DATA_PROCESSOR_CLS = {
    'None': None,
    'BaseDataProcessor': BaseDataProcessor,
    'SmartKBDataProcessor': SmartKBDataProcessor,
}


def build_common_parser():
    parser = argparse.ArgumentParser(description="Common arguments for training pointwise intent prediction model")

    # Common arguments
    parser.add_argument("--grass_region", type=str, default="sg", help="Modelling region")
    parser.add_argument("--feature_config_path", type=str, required=True, help="Path to user & intent feature config file")
    parser.add_argument("--model_config_path", type=str, required=True, help="Path to user & intent model config file")
    parser.add_argument("--user_feature_dict_path", type=str, required=True, help="Path to user feature dict file")
    parser.add_argument("--user_data_path", type=str, required=True, help="Path to user data folder")
    parser.add_argument("--user_data_path_for_valid", type=str, default="", help="Path to user data folder only for validation")
    parser.add_argument("--intent_feature_dict_path", type=str, required=True, help="Path to intent feature dict file")
    parser.add_argument("--intent_data_path", type=str, required=True, help="Path to intent data folder")
    parser.add_argument("--label_dict_path", type=str, required=True, help="Path to label dict file")
    parser.add_argument("--text_embedding_path", type=str, default="", help="Path to text embedding file")
    parser.add_argument("--intent_sid_path", type=str, default="", help="Path to intent sid file")
    parser.add_argument("--intent_sid_fids", type=str, default="", help="Intent SID mapped FIDs, split by ',' e.g.'1234,2345,3456'")
    parser.add_argument("--train_start_date", type=str, default="2024-05-01", help="Training start date")
    parser.add_argument("--train_end_date", type=str, default="2024-05-12", help="Training end date")
    parser.add_argument("--valid_start_date", type=str, default="2024-05-13", help="Valid start date")
    parser.add_argument("--valid_end_date", type=str, default="2024-05-14", help="Valid end date")
    parser.add_argument("--data_processor", type=str, default='BaseDataProcessor', help="class name of data processor, pass `None` to skip data processor")
    parser.add_argument("--data_version", type=str, default='01', help='output version name of processed data')
    parser.add_argument("--model_type", type=str, default='IPFusionDebiasModel', help='class name of model type')
    parser.add_argument("--model_prefix_path", type=str, required=True, help="Absolute path prefix to store user model result")
    parser.add_argument("--model_result", type=str, required=True, help="Path postfix for model result")
    parser.add_argument("--model_output", type=str, required=True, help="Path postfix for model output")
    parser.add_argument("--model_resume_training", type=bool, default=False, help="Whether to resume training or not")
    parser.add_argument("--wandb_project", type=str, default="intent_prediction", help="wandb project name for training log")
    parser.add_argument("--best_model_metric", type=str, default="recall_5", help="Best model metric for selection")
    parser.add_argument("--enable_epoch_logging", action='store_true', default=False, help="Whether to enable logging metrics at the end of each epoch")

    # Smart KB arguments
    parser.add_argument("--intent_issue_mapping_path", type=str, default="", help="Path to intent issue mapping file")
    parser.add_argument("--smartkb_abtest_groups", type=str, default="", help="Smart KB abtest groups, split by ','")

    return parser


def parse_training_args(parser):
    # Common arguments
    args, remaining_args = parser.parse_known_args()
    args.grass_region = args.grass_region.lower()
    args.model_result_path = os.path.join(args.model_prefix_path, args.model_result)
    args.model_output_path = os.path.join(args.model_prefix_path, args.model_output)

    # Model arguments
    # Different model type have different parameters,
    # all model parameter names should start with 'model.' when specifying arguments.
    model_parser = argparse.ArgumentParser()

    assert(args.data_processor in DATA_PROCESSOR_CLS)

    assert(args.model_type in MODEL_CLS_DICT)
    if args.model_type in ("IPFusionModel", "IPFusionDebiasModel", "IPBiFusionDebiasModel"):
        model_parser.add_argument("--model.emb_size", type=int, default=256)
        model_parser.add_argument("--model.dropout", type=float, default=0.2)

    elif args.model_type in ("IPDCNv2Model", "IPFusionDCNv2Model", "IPDCNv2DebiasModel"):
        model_parser.add_argument("--model.num_layers", type=int, default=3)
        model_parser.add_argument("--model.emb_size", type=int, default=256)
        model_parser.add_argument("--model.dropout", type=float, default=0.2)

    elif args.model_type in ("IPFusion3TowerModel", "IPFusionDebias3TowerModel"):
        model_parser.add_argument("--model.emb_size", type=int, default=256)
        model_parser.add_argument("--model.dropout", type=float, default=0.2)
        model_parser.add_argument("--model.use_uq_gate", type=bool, default=False)
        model_parser.add_argument("--model.uq_weight", type=float, default=0.7)

    elif args.model_type in (
            "IPFusionRankMixerModel",
            "IPFusionGroupRankMixerModel",
            "IPFusionDebiasUserRankMixerModel",
            "IPFusionDebiasGroupRankMixerModel",
            "IPFusionGroupTokenMixerModel",
            "IPFusionGroupRankMixerDebiasModel",
        ):
        model_parser.add_argument("--model.emb_size", type=int, default=256)
        model_parser.add_argument("--model.dropout", type=float, default=0.2)
        model_parser.add_argument("--model.num_layers", type=int, default=2)
        model_parser.add_argument("--model.ffn_scale_ratio", type=int, default=3)
        model_parser.add_argument("--model.ffn_type", type=str, default="per_token_dense", choices=["share_all", "per_token_dense", "per_token_sparse_moe"])
        model_parser.add_argument("--model.num_experts", type=int, default=3)
        model_parser.add_argument("--model.ffn_ln", action='store_true', default=False)
        model_parser.add_argument("--model.mix_ln", action='store_true', default=False)

        if args.model_type in ("IPFusionGroupTokenMixerModel"):
            model_parser.add_argument("--model.num_heads", type=int, default=None)

    else:
        raise ValueError(f"model_type not supported: {args.model_type}")

    model_args = model_parser.parse_args(remaining_args)
    model_kwargs = {
        k.split(".", 1)[1]: v
        for k, v in vars(model_args).items()
        if k.startswith("model.")
    }
    return args, model_kwargs
