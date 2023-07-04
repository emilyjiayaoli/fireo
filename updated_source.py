def forward(self, rgb_input, ir_input):
    
    rgb_z = self.rgb_encoder(rgb_input)
    rgb_z = rgb_z.view(rgb_z.size(0), -1)

    ir_z = self.ir_encoder(ir_input)
    ir_z = ir_z.view(ir_z.size(0), -1)

    z = torch.cat((rgb_z, ir_z), dim=-1)

    return locals(),  self.projection(z)