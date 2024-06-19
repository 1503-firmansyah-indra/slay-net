from datetime import datetime
import os

from loguru import logger


class CheckpointManager:
    def __init__(self, args,
                 combined_epochs_done=0, contrastive_epochs_done=0, compatibility_epochs_done=0):
        '''

        :param args:
        '''
        self.chpt_name_modifier_combined = combined_epochs_done
        self.chpt_name_modifier_contrastive = contrastive_epochs_done
        self.chpt_name_modifier_comp = compatibility_epochs_done

        self.max_attempt = 10
        self.config_file_name = args.config_dir.split('/')[-1].split('.yaml')[0]

    def generate_name(self, learning_type, checkpoint_dir, current_epoch_index, instance_count):
        '''
        This method ensures that the name of the checkpoint has not already been taken
        :param learning_type:
        :param checkpoint_dir:
        :param current_epoch_index:
        :param instance_count:
        :return:
        '''
        assert learning_type in ['combined', 'contrastive', 'compatibility']
        contrastive_epochs = self.chpt_name_modifier_contrastive + current_epoch_index + 1 if learning_type == 'contrastive' \
            else self.chpt_name_modifier_contrastive
        comp_epochs = self.chpt_name_modifier_comp + current_epoch_index + 1 if learning_type == 'compatibility' \
            else self.chpt_name_modifier_comp
        combined_epochs = self.chpt_name_modifier_combined + current_epoch_index + 1 if learning_type == 'combined' \
            else self.chpt_name_modifier_combined

        name_affix = f"{self.config_file_name}_cbe{combined_epochs}_te{contrastive_epochs}_ce{comp_epochs}" + \
            f"_inst{instance_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        checkpoint_path = os.path.join(checkpoint_dir, f"{name_affix}.pth")

        unique_name_found = False
        for attempt_i in range(self.max_attempt):
            if not os.path.exists(checkpoint_path):
                unique_name_found = True
                break
            name_affix = f"{self.config_file_name}_cbe{combined_epochs}_te{contrastive_epochs}_ce{comp_epochs}" + \
                         f"_inst{instance_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{attempt_i}"
            checkpoint_path = os.path.join(checkpoint_dir, f"{name_affix}.pth")
        if not unique_name_found:
            name_affix = f"{self.config_file_name}_cbe{combined_epochs}_te{contrastive_epochs}_ce{comp_epochs}" + \
                f"_inst{instance_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_x"
            checkpoint_path = os.path.join(checkpoint_dir, f"{name_affix}.pth")
        return checkpoint_path


class CheckpointMetricManager:
    def __init__(self, metric_type):
        assert metric_type in ['compatibility_auc', 'fitb_accuracy']
        self.metric_type = metric_type
        self.epoch_to_calculated_metrics = {}
        self.best_metric_value = 0
        self.best_epoch = None

    def add_epoch_metric(self, epoch_number, value, checkpoint_path):
        assert epoch_number not in self.epoch_to_calculated_metrics.keys()
        self.epoch_to_calculated_metrics[epoch_number] = {
            "metric_type": self.metric_type,
            "value": value,
            "checkpoint_path": checkpoint_path
        }
        if value >= self.best_metric_value:
            logger.info(f"Checkpoint metric | best epoch is updated to '{epoch_number}' with metric value '{value}'")
            self.best_metric_value = value
            self.best_epoch = epoch_number
        return True

    def get_best_epoch_metric(self):
        if self.best_epoch is not None:
            logger.info(f"Checkpoint metric | best epoch is '{self.best_epoch}' "
                        f"with metric '{self.metric_type}' of '{self.best_metric_value}'")
            return self.epoch_to_calculated_metrics[self.best_epoch]
        else:
            logger.info("Checkpoint metric | No epoch has been completed, None is returned")
            return None

