from datetime import datetime
import os


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
